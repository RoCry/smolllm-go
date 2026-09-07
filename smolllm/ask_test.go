package smolllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAskUsesReasoningEffortOption(t *testing.T) {
	t.Parallel()

	var captured map[string]any
	var decodeErr error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeErr = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		WithReasoningEffort("none"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireAnswered(t, msg)
	require.NoError(t, decodeErr)

	assert.Equal(t, "hello", msg.Content)
	assert.Equal(t, "openai/gpt-5", msg.Model)
	assert.Equal(t, "gpt-5", msg.ModelName)
	assert.Equal(t, "gpt-5", captured["model"])
	assert.Equal(t, "none", captured["reasoning_effort"])
}

func TestAskUsesProviderStreamingUsage(t *testing.T) {
	t.Parallel()

	usageFrame := fmt.Sprintf(
		"data: %s\n\n",
		`{"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`,
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte(usageFrame))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireAnswered(t, msg)

	assert.Equal(t, "hello", msg.Content)
	assert.Equal(t, 11, msg.Usage.Input)
	assert.Equal(t, 7, msg.Usage.Output)
	assert.False(t, msg.Usage.Estimated)
}

func TestAskRetriesWithoutStreamOptionsWhenProviderRejectsIt(t *testing.T) {
	t.Parallel()

	var requestCount int
	var retryPayload map[string]any
	var decodeErr error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			decodeErr = err
			http.Error(w, "decode request", http.StatusBadRequest)
			return
		}
		if requestCount == 1 {
			assert.Contains(t, payload, "stream_options")
			http.Error(w, "unknown field stream_options", http.StatusBadRequest)
			return
		}
		retryPayload = payload
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireAnswered(t, msg)
	require.NoError(t, decodeErr)

	assert.Equal(t, "hello", msg.Content)
	assert.Equal(t, 2, requestCount)
	assert.NotContains(t, retryPayload, "stream_options")
	assert.True(t, msg.Usage.Estimated)
}

func TestAskDoesNotRetryWithoutStreamOptionsOnRateLimit(t *testing.T) {
	t.Parallel()

	var requestCount int
	var decodeErr error
	var firstPayload map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			decodeErr = err
			http.Error(w, "decode request", http.StatusBadRequest)
			return
		}
		if requestCount == 1 {
			firstPayload = payload
		}
		http.Error(w, "rate limited", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireFailed(t, msg)
	require.NoError(t, decodeErr)
	assert.Equal(t, 1, requestCount)
	assert.Contains(t, firstPayload, "stream_options")
}

func TestAskKeepsOriginalBadRequestBodyWhenStreamOptionsRetryTransportFails(t *testing.T) {
	t.Parallel()

	client := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			var payload map[string]any
			require.NoError(t, json.NewDecoder(req.Body).Decode(&payload))
			if _, ok := payload["stream_options"]; ok {
				return testHTTPResponse(req, http.StatusBadRequest, "unknown field stream_options"), nil
			}
			return nil, errors.New("network down")
		}),
		CheckRedirect: nil,
		Jar:           nil,
		Timeout:       0,
	}

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider("https://example.test/", "test-key"),
		WithHTTPClient(client),
	)
	requireFailed(t, msg)
	assert.Contains(t, msg.ErrorMessage, "unknown field stream_options")
	assert.Contains(t, msg.ErrorMessage, "network down")
}

func TestAskRecordsEveryAttemptOnTheMessage(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer srv.Close()

	var hooked []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithHook(func(attempt Attempt) {
			hooked = append(hooked, attempt)
		}),
	)
	requireFailed(t, msg)

	// The hook fires live and Attempts collects the same data at the end.
	require.Len(t, hooked, 1)
	require.Len(t, msg.Attempts, 1)
	require.NotNil(t, hooked[0].Err)
	assert.True(t, hooked[0].Failed())
	assert.Equal(t, "openai", hooked[0].Provider)
	assert.Equal(t, "openai/gpt-5", hooked[0].Model)
	assert.Equal(t, 0, hooked[0].Usage.Output)
	assert.True(t, hooked[0].Usage.Estimated)
	assert.Equal(t, hooked[0].Model, msg.Attempts[0].Model)
}

func TestAskAttemptKeepsReportedUsageWhenAGuardRejectsTheAnswer(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		usageFrame := `data: {"choices":[],"usage":{"prompt_tokens":1234,` +
			`"completion_tokens":2,"total_tokens":1236}}` + "\n\n"
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		_, _ = w.Write([]byte(usageFrame))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	var hooked []Attempt
	msg := Ask(context.Background(), RequestFromString(strings.Repeat("prompt ", 5000)),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithMinOutputTokens(5),
		WithHook(func(attempt Attempt) {
			hooked = append(hooked, attempt)
		}),
	)
	requireFailed(t, msg)

	require.Len(t, hooked, 1)
	require.NotNil(t, hooked[0].Err)
	assert.True(t, hooked[0].Failed())
	// The tokens were spent even though the answer was rejected.
	assert.Equal(t, 1234, hooked[0].Usage.Input)
	assert.Equal(t, 2, hooked[0].Usage.Output)
	assert.False(t, hooked[0].Usage.Estimated)
	// The live hook and the collected list report the same attempt.
	require.Len(t, msg.Attempts, 1)
	assert.Equal(t, hooked[0].Usage, msg.Attempts[0].Usage)
}

func TestAskAttemptKeepsReportedUsageWhenResponseIsEmpty(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		usageFrame := `data: {"choices":[],"usage":{"prompt_tokens":12,` +
			`"completion_tokens":0,"total_tokens":12}}` + "\n\n"
		_, _ = w.Write([]byte(usageFrame))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	var hooked []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithHook(func(attempt Attempt) {
			hooked = append(hooked, attempt)
		}),
	)
	requireFailed(t, msg)

	require.Len(t, hooked, 1)
	require.NotNil(t, hooked[0].Err)
	assert.True(t, hooked[0].Failed())
	assert.Equal(t, 12, hooked[0].Usage.Input)
	assert.Equal(t, 0, hooked[0].Usage.Output)
	assert.False(t, hooked[0].Usage.Estimated)
	// The live hook and the collected list report the same attempt.
	require.Len(t, msg.Attempts, 1)
	assert.Equal(t, hooked[0].Usage, msg.Attempts[0].Usage)
}

func TestAskReportsMalformedRequestWithoutAttemptingALeg(t *testing.T) {
	t.Parallel()

	var reached bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		reached = true
		writeChatSuccess(t, w, "hello", "stop")
	}))
	defer srv.Close()

	// Malformed input is data, not a coding contract violation, so it ends the
	// call as a terminal error rather than panicking.
	msg := Ask(context.Background(), Request{System: "", Messages: nil, Tools: nil},
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireFailed(t, msg)
	assert.Contains(t, msg.ErrorMessage, "at least one message")
	assert.False(t, reached, "a malformed request must never reach a provider")
	assert.Empty(t, msg.Attempts)
}

func TestStreamPanicsOnNilContext(t *testing.T) {
	t.Parallel()

	// A nil context is a programmer error, which panics rather than becoming a
	// terminal message.
	require.PanicsWithValue(t, "smolllm: context must not be nil", func() {
		//nolint:staticcheck // passing nil is exactly what this test pins
		_ = Stream(nil, RequestFromString("hi"), WithModel("openai/gpt-5"))
	})
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func testHTTPResponse(req *http.Request, statusCode int, body string) *http.Response {
	rec := httptest.NewRecorder()
	rec.WriteHeader(statusCode)
	_, _ = rec.WriteString(body)
	resp := rec.Result()
	resp.Request = req
	return resp
}
