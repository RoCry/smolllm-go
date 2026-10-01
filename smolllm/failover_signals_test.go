package smolllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testCacheKey = "conv-42"

// fallbackServer answers model-b with a plain success and hands model-a to
// primary, counting how often each was asked.
func fallbackServer(t *testing.T, primary http.HandlerFunc) (*httptest.Server, *atomic.Int32, *atomic.Int32) {
	t.Helper()
	var primaryCalls, fallbackCalls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		switch model {
		case testModelA:
			primaryCalls.Add(1)
			primary(w, r)
		case testModelB:
			fallbackCalls.Add(1)
			writeChatSuccess(t, w, "fallback answer", "stop")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &primaryCalls, &fallbackCalls
}

func TestAskAdvancesWhenGatewaySaysDoNotRetry(t *testing.T) {
	t.Parallel()
	srv, primary, fallback := fallbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(headerShouldRetry, "false")
		http.Error(w, `{"error":{"message":"codex is unreachable"}}`, http.StatusBadGateway)
	})

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "fallback answer", msg.Content)
	assert.Equal(t, int32(1), primary.Load(), "a gateway that already retried is not retried again")
	assert.Equal(t, int32(1), fallback.Load())
}

func TestAskFailsLegOnInBandStreamError(t *testing.T) {
	t.Parallel()
	srv, _, fallback := fallbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeFakeResponse(t, w,
			"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"error\"}],"+
				"\"error\":{\"message\":\"upstream died\",\"type\":\"upstream_error\"}}\n\n",
			"data: [DONE]\n\n",
		)
	})

	var events []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithHook(func(event Attempt) { events = append(events, event) }),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "fallback answer", msg.Content, "the failed leg's partial text never reaches the answer")
	assert.Equal(t, int32(1), fallback.Load())
	require.Len(t, events, 2)
	require.NotNil(t, events[0].Err)
	assert.Contains(t, events[0].Err.Error(), "upstream died")
	assert.True(t, events[0].Emitted, "the leg had produced text")
	assert.Equal(t, DispositionAdvance, events[0].Err.Disposition, "Ask emits nothing early, so it never commits")
}

func TestAskReportsInBandStringError(t *testing.T) {
	t.Parallel()
	srv, _, _ := fallbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeFakeResponse(t, w, "data: {\"error\":\"overloaded\"}\n\n")
	})

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	assert.Equal(t, StopReasonError, msg.StopReason)
	assert.Contains(t, msg.ErrorMessage, "overloaded")
	require.Len(t, msg.Attempts, 1)
	assert.Equal(t, time.Duration(-1), msg.Attempts[0].TTFT, "the stream was read but no token arrived")
}

func TestAskLegBudgetAdvancesPastAHungProvider(t *testing.T) {
	t.Parallel()
	srv, primary, fallback := fallbackServer(t, func(_ http.ResponseWriter, r *http.Request) {
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	})

	started := time.Now()
	var events []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithLegBudget(200*time.Millisecond),
		WithHook(func(event Attempt) { events = append(events, event) }),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "fallback answer", msg.Content)
	assert.Less(t, time.Since(started), 3*time.Second, "the hung leg costs its budget, not the provider's patience")
	assert.Equal(t, int32(1), primary.Load(), "a spent budget is not retried")
	assert.Equal(t, int32(1), fallback.Load())
	require.NotNil(t, events[0].Err)
	require.ErrorIs(t, events[0].Err, ErrLegBudgetExceeded)
}

func TestAskLegBudgetStopsOnceTheLegStreams(t *testing.T) {
	t.Parallel()
	srv, _, fallback := fallbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeFakeResponse(t, w, "data: {\"choices\":[{\"delta\":{\"content\":\"slow \"}}]}\n\n")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Errorf("httptest writer cannot flush")
			return
		}
		flusher.Flush()
		time.Sleep(400 * time.Millisecond)
		writeFakeResponse(t, w,
			"data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n",
			"data: [DONE]\n\n",
		)
	})

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithLegBudget(100*time.Millisecond),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "slow answer", msg.Content)
	assert.Equal(t, int32(0), fallback.Load())
}

func TestProviderExtraBodyReachesOnlyThatProvider(t *testing.T) {
	t.Parallel()
	bodies := make(chan map[string]any, 2)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode fake provider request: %v", err)
		}
		bodies <- payload
		if payload["model"] == testModelA {
			http.Error(w, "quota", http.StatusTooManyRequests)
			return
		}
		writeChatSuccess(t, w, "ok", "stop")
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithExtraBody(map[string]any{"user": "chain-wide"}),
		WithProviderExtraBody(providerOpenAI, map[string]any{"prompt_cache_key": testCacheKey, "user": "u1"}),
	)
	requireAnswered(t, msg)

	first, second := <-bodies, <-bodies
	assert.Equal(t, testCacheKey, first["prompt_cache_key"])
	assert.Equal(t, "u1", first["user"], "the provider's own fields win over the chain-wide ones")
	assert.NotContains(t, second, "prompt_cache_key")
	assert.Equal(t, "chain-wide", second["user"])
}

func TestClassifyHonorsShouldRetryFalse(t *testing.T) {
	t.Parallel()
	assert.Equal(t, DispositionAdvance,
		Classify(&HTTPError{StatusCode: http.StatusBadGateway, Body: "x", NoRetry: true}))
	assert.Equal(t, DispositionAbort,
		Classify(&HTTPError{StatusCode: http.StatusBadRequest, Body: "x", NoRetry: true}),
		"a malformed request stays chain-global whatever the header says")
}
