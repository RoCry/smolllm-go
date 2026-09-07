package smolllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Wire model names the fake providers in this file switch on.
const (
	testModelA = "model-a"
	testModelB = "model-b"
)

func TestAskRetriesServerErrorsBeforeFallingBack(t *testing.T) {
	t.Parallel()
	const expectedAttempts = 3 // Initial call plus the two configured retries.

	var firstProviderAttempts atomic.Int32
	var fallbackAttempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		switch model {
		case testModelA:
			firstProviderAttempts.Add(1)
			http.Error(w, "upstream unavailable", http.StatusInternalServerError)
		case testModelB:
			fallbackAttempts.Add(1)
			writeChatSuccess(t, w, "fallback answer", "length")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer srv.Close()

	var events []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithHook(func(event Attempt) {
			events = append(events, event)
		}),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "fallback answer", msg.Content)
	assert.Equal(t, "length", msg.FinishReason)
	assert.Equal(t, StopReasonLength, msg.StopReason)
	assert.Equal(t, int32(expectedAttempts), firstProviderAttempts.Load())
	assert.Equal(t, int32(1), fallbackAttempts.Load())
	require.Len(t, events, expectedAttempts+1)
	for _, event := range events[:expectedAttempts] {
		assert.Equal(t, "openai", event.Provider)
		assert.Equal(t, "openai/model-a", event.Model)
		require.NotNil(t, event.Err)
	}
	assert.Equal(t, "gemini", events[expectedAttempts].Provider)
	assert.Equal(t, "gemini/model-b", events[expectedAttempts].Model)
	assert.Nil(t, events[expectedAttempts].Err)
}

func TestAskFallsBackImmediatelyOnRateLimit(t *testing.T) {
	t.Parallel()

	var firstProviderAttempts atomic.Int32
	var fallbackAttempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		switch model {
		case testModelA:
			firstProviderAttempts.Add(1)
			http.Error(w, "quota exhausted", http.StatusTooManyRequests)
		case testModelB:
			fallbackAttempts.Add(1)
			writeChatSuccess(t, w, "fallback answer", "provider-specific")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer srv.Close()

	var events []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithHook(func(event Attempt) {
			events = append(events, event)
		}),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "fallback answer", msg.Content)
	// A provider string smolllm does not know still reaches the caller verbatim,
	// normalized to the ordinary stop reason.
	assert.Equal(t, "provider-specific", msg.FinishReason)
	assert.Equal(t, StopReasonStop, msg.StopReason)
	assert.Equal(t, int32(1), firstProviderAttempts.Load())
	assert.Equal(t, int32(1), fallbackAttempts.Load())
	require.Len(t, events, 2)
	assert.Equal(t, "openai", events[0].Provider)
	assert.Equal(t, "openai/model-a", events[0].Model)
	require.NotNil(t, events[0].Err)
	assert.Equal(t, "gemini", events[1].Provider)
	assert.Equal(t, "gemini/model-b", events[1].Model)
	assert.Nil(t, events[1].Err)
}

func TestAskFallsBackOnTruncatedEmptyContent(t *testing.T) {
	t.Parallel()

	var firstProviderAttempts atomic.Int32
	var fallbackAttempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		switch model {
		case testModelA:
			firstProviderAttempts.Add(1)
			writeChatTruncatedReasoning(t, w)
		case testModelB:
			fallbackAttempts.Add(1)
			writeChatSuccess(t, w, "fallback answer", "stop")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer srv.Close()

	var events []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithHook(func(event Attempt) {
			events = append(events, event)
		}),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "fallback answer", msg.Content)
	// Deterministic truncation must not burn same-leg retries: one attempt each.
	assert.Equal(t, int32(1), firstProviderAttempts.Load())
	assert.Equal(t, int32(1), fallbackAttempts.Load())
	require.Len(t, events, 2)
	assert.Equal(t, "openai/model-a", events[0].Model)
	require.NotNil(t, events[0].Err)
	assert.Contains(t, events[0].Err.Error(), "truncated before any content")
	assert.Equal(t, "gemini/model-b", events[1].Model)
	assert.Nil(t, events[1].Err)
}

func TestAskSurfacesTruncatedEmptyContentOnLastLeg(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeChatTruncatedReasoning(t, w)
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireFailed(t, msg)
	assert.Contains(t, msg.ErrorMessage, "truncated before any content")
}

// A chain that exhausts every leg must explain every leg. Reporting only the
// last one hides why the earlier candidates were skipped.
func TestAskErrorNamesEveryFailedLeg(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		switch model {
		case testModelA:
			http.Error(w, "quota exhausted", http.StatusTooManyRequests)
		case testModelB:
			http.Error(w, "key revoked", http.StatusUnauthorized)
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireFailed(t, msg)
	require.Len(t, msg.Attempts, 2, "every leg is recorded, not only the last")

	message := msg.ErrorMessage
	assert.Contains(t, message, "openai/model-a")
	assert.Contains(t, message, "quota exhausted")
	assert.Contains(t, message, "gemini/model-b")
	assert.Contains(t, message, "key revoked")
}

func TestAskAbortsChainOnMalformedRequestStatus(t *testing.T) {
	t.Parallel()

	// A 400 says the request shape is wrong, which is true for every leg, so the
	// chain must stop instead of spending the second provider's quota.
	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		http.Error(w, "unsupported parameter", http.StatusBadRequest)
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireFailed(t, msg)
	// Two requests for the one leg: the original, plus the stream_options probe
	// every 400 triggers. A chain that advanced would double that.
	assert.Equal(t, int32(2), attempts.Load(), "an aborting failure must not reach the second leg")
	assert.NotContains(t, msg.ErrorMessage, "gemini/model-b")
}

func requestModel(r *http.Request) (string, error) {
	var payload struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		return "", fmt.Errorf("decode request: %w", err)
	}
	return payload.Model, nil
}

// writeChatTruncatedReasoning emulates a thinking model whose reasoning consumed
// the entire output budget: reasoning deltas only, no content, finish_reason=length.
func writeChatTruncatedReasoning(t *testing.T, w http.ResponseWriter) {
	t.Helper()

	w.Header().Set("Content-Type", "text/event-stream")
	_, err := fmt.Fprint(w,
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking hard about the answer\"}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"+
			"data: [DONE]\n\n",
	)
	if err != nil {
		t.Errorf("write fake provider response: %v", err)
		return
	}
}

func writeChatSuccess(t *testing.T, w http.ResponseWriter, answer, finishReason string) {
	t.Helper()

	w.Header().Set("Content-Type", "text/event-stream")
	_, err := fmt.Fprintf(w,
		"data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":%q}]}\n\n"+
			"data: [DONE]\n\n",
		answer,
		finishReason,
	)
	if err != nil {
		t.Errorf("write fake provider response: %v", err)
		return
	}
}
