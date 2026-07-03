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

func TestAskUsesReasoningEffortSuffix(t *testing.T) {
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

	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5!none"),
		WithReasoningEffort("medium"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	require.NoError(t, decodeErr)

	assert.Equal(t, "hello", resp.Text)
	assert.Equal(t, "openai/gpt-5", resp.Model)
	assert.Equal(t, "gpt-5", resp.ModelName)
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

	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	assert.Equal(t, "hello", resp.Text)
	assert.Equal(t, 11, resp.Usage.InputTokens)
	assert.Equal(t, 7, resp.Usage.OutputTokens)
	assert.False(t, resp.Usage.Estimated)
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

	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	require.NoError(t, decodeErr)

	assert.Equal(t, "hello", resp.Text)
	assert.Equal(t, 2, requestCount)
	assert.NotContains(t, retryPayload, "stream_options")
	assert.True(t, resp.Usage.Estimated)
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

	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.Error(t, err)
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

	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL("https://example.test/"),
		WithAPIKey("test-key"),
		WithHTTPClient(client),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unknown field stream_options")
	assert.Contains(t, err.Error(), "network down")
}

func TestAskHookFiresOnHTTPFailure(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer srv.Close()

	var events []RequestEvent
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithHook(func(event RequestEvent) {
			events = append(events, event)
		}),
	)
	require.Error(t, err)
	require.Len(t, events, 1)
	require.Error(t, events[0].Error)
	assert.Equal(t, "openai", events[0].Provider)
	assert.Equal(t, "openai/gpt-5", events[0].Model)
	assert.Equal(t, 0, events[0].OutputTokens)
	assert.True(t, events[0].Estimated)
}

func TestAskFailureHookUsesReportedUsageWhenPostSuccessValidationFails(t *testing.T) {
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

	var events []RequestEvent
	_, err := Ask(context.Background(), PromptFromString(strings.Repeat("prompt ", 5000)),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithMinOutputTokens(5),
		WithHook(func(event RequestEvent) {
			events = append(events, event)
		}),
	)
	require.Error(t, err)
	require.Len(t, events, 1)
	require.Error(t, events[0].Error)
	assert.Equal(t, 1234, events[0].InputTokens)
	assert.Equal(t, 2, events[0].OutputTokens)
	assert.False(t, events[0].Estimated)
}

func TestAskFailureHookUsesReportedUsageWhenEmptyResponseFails(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		usageFrame := `data: {"choices":[],"usage":{"prompt_tokens":12,` +
			`"completion_tokens":0,"total_tokens":12}}` + "\n\n"
		_, _ = w.Write([]byte(usageFrame))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	var events []RequestEvent
	_, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithHook(func(event RequestEvent) {
			events = append(events, event)
		}),
	)
	require.Error(t, err)
	require.Len(t, events, 1)
	require.Error(t, events[0].Error)
	assert.Equal(t, 12, events[0].InputTokens)
	assert.Equal(t, 0, events[0].OutputTokens)
	assert.False(t, events[0].Estimated)
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
