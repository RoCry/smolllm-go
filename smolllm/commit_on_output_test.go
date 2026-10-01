package smolllm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// failAfter streams the given frames from model-a, then reports an in-band
// error, which on its own the chain classifies as advance.
func failAfter(t *testing.T, frames ...string) http.HandlerFunc {
	t.Helper()
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeFakeResponse(t, w, frames...)
		writeFakeResponse(t, w, "data: {\"error\":{\"message\":\"upstream died\"}}\n\n")
	}
}

// requireCommitted checks a Stream that a leg's emitted output pinned to
// model-a: a terminal error naming that leg, and nothing from model-b.
func requireCommitted(t *testing.T, events []Event, msg *AssistantMessage, fallbackCalls int32) {
	t.Helper()
	requireFailed(t, msg)
	assert.Equal(t, int32(0), fallbackCalls, "the next leg is never started")
	assert.Contains(t, msg.ErrorMessage, `leg "openai/model-a"`)
	assert.Contains(t, msg.ErrorMessage, "upstream died")
	assert.Contains(t, msg.ErrorMessage, "output already emitted, so the chain did not advance")
	assert.Equal(t, "openai/model-a", msg.Model)

	require.Len(t, msg.Attempts, 1)
	attempt := msg.Attempts[0]
	assert.True(t, attempt.Emitted)
	assert.GreaterOrEqual(t, attempt.TTFT, time.Duration(0), "a failed attempt keeps its real TTFT")
	require.NotNil(t, attempt.Err)
	assert.Equal(t, DispositionAbort, attempt.Err.Disposition)

	failed := eventsOfKind(events, EventLegFailed)
	require.Len(t, failed, 1)
	require.NotNil(t, failed[0].Attempt)
	assert.True(t, failed[0].Attempt.Emitted)
	require.GreaterOrEqual(t, len(events), 2)
	assert.Equal(t, EventError, events[len(events)-1].Kind)
	assert.Equal(t, EventLegFailed, events[len(events)-2].Kind, "nothing follows the failure but the terminal event")
}

func TestStreamFailureAfterTextEndsTheCall(t *testing.T) {
	t.Parallel()
	srv, _, fallback := fallbackServer(t, failAfter(t,
		"data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"partial\\\":\"}}]}\n\n"))

	events, msg := collect(Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireCommitted(t, events, msg, fallback.Load())
	assert.Equal(t, `{"partial":`, deltaText(events, EventTextDelta), "only the committed leg's text was emitted")
}

func TestStreamFailureAfterToolCallFragmentEndsTheCall(t *testing.T) {
	t.Parallel()
	srv, _, fallback := fallbackServer(t, failAfter(t,
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\","+
			"\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"ci\"}}]}}]}\n\n"))

	events, msg := collect(Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireCommitted(t, events, msg, fallback.Load())
	assert.NotEmpty(t, eventsOfKind(events, EventToolCallDelta))
	assert.Empty(t, eventsOfKind(events, EventToolCallEnd))
}

// A failed turn names the last leg tried, not the last one that got a 200.
func TestFailedTurnNamesTheLastLegTried(t *testing.T) {
	t.Parallel()
	primary := failAfter(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			return
		}
		if model == testModelA {
			primary(w, r)
			return
		}
		http.Error(w, "quota", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	msg := drain(Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireFailed(t, msg)
	assert.Equal(t, "gemini/model-b", msg.Model)
	assert.Equal(t, providerGemini, msg.Provider)
}
