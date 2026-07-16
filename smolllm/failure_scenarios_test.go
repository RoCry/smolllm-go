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
		case "model-a":
			firstProviderAttempts.Add(1)
			http.Error(w, "upstream unavailable", http.StatusInternalServerError)
		case "model-b":
			fallbackAttempts.Add(1)
			writeChatSuccess(t, w, "fallback answer", "length")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer srv.Close()

	var events []RequestEvent
	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithHook(func(event RequestEvent) {
			events = append(events, event)
		}),
	)
	require.NoError(t, err)
	assert.Equal(t, "fallback answer", resp.Text)
	assert.Equal(t, "length", resp.FinishReason)
	assert.Equal(t, int32(expectedAttempts), firstProviderAttempts.Load())
	assert.Equal(t, int32(1), fallbackAttempts.Load())
	require.Len(t, events, expectedAttempts+1)
	for _, event := range events[:expectedAttempts] {
		assert.Equal(t, "openai", event.Provider)
		assert.Equal(t, "openai/model-a", event.Model)
		require.Error(t, event.Error)
	}
	assert.Equal(t, "gemini", events[expectedAttempts].Provider)
	assert.Equal(t, "gemini/model-b", events[expectedAttempts].Model)
	assert.NoError(t, events[expectedAttempts].Error)
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
		case "model-a":
			firstProviderAttempts.Add(1)
			http.Error(w, "quota exhausted", http.StatusTooManyRequests)
		case "model-b":
			fallbackAttempts.Add(1)
			writeChatSuccess(t, w, "fallback answer", "provider-specific")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer srv.Close()

	var events []RequestEvent
	resp, err := Ask(context.Background(), PromptFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
		WithHook(func(event RequestEvent) {
			events = append(events, event)
		}),
	)
	require.NoError(t, err)
	assert.Equal(t, "fallback answer", resp.Text)
	assert.Equal(t, "provider-specific", resp.FinishReason)
	assert.Equal(t, int32(1), firstProviderAttempts.Load())
	assert.Equal(t, int32(1), fallbackAttempts.Load())
	require.Len(t, events, 2)
	assert.Equal(t, "openai", events[0].Provider)
	assert.Equal(t, "openai/model-a", events[0].Model)
	require.Error(t, events[0].Error)
	assert.Equal(t, "gemini", events[1].Provider)
	assert.Equal(t, "gemini/model-b", events[1].Model)
	assert.NoError(t, events[1].Error)
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
