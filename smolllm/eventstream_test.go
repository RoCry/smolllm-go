package smolllm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newSlowStreamServer dribbles chunks so a test can act while a call is still
// in flight. release closes when the handler is done.
func newSlowStreamServer(t *testing.T, chunks int, gap time.Duration) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for i := range chunks {
			_, err := fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"%d\"}}]}\n\n", i)
			if err != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
			select {
			case <-r.Context().Done():
				return
			case <-time.After(gap):
			}
		}
		_, _ = fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
}

func TestEventStreamDeliversEventsInOrder(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireAnswered(t, msg)

	require.NotEmpty(t, events)
	assert.Equal(t, EventStart, events[0].Kind, "start comes first")
	assert.Equal(t, EventDone, events[len(events)-1].Kind, "a terminal event comes last")
	for _, event := range events[:len(events)-1] {
		assert.False(t, event.Kind.Terminal(), "only the last event is terminal")
	}
	assert.Equal(t, "hello", deltaText(events, EventTextDelta))
}

// Every event carries a snapshot, and a snapshot never changes afterwards.
func TestEventStreamSnapshotsAreImmutable(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireAnswered(t, msg)

	require.NotEmpty(t, events)
	for _, event := range events {
		require.NotNil(t, event.Message, "every event carries a snapshot")
	}

	first := events[0].Message
	assert.Empty(t, first.Content, "the start snapshot has no text yet")
	assert.Equal(t, StopReasonPending, first.StopReason)
	assert.False(t, first.StopReason.Terminal())
	// The terminal snapshot has the whole answer, and the earlier one is unchanged.
	assert.Equal(t, "hello", msg.Content)
	assert.Empty(t, first.Content)
}

func TestEventStreamResultWithoutDrainingEvents(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	stream := Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	// The unbounded queue means the call finishes even with nobody reading.
	msg := stream.Result()
	requireAnswered(t, msg)
	assert.Equal(t, "hello", msg.Content)

	stream.Close()
}

func TestEventStreamResultIsIdempotent(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	stream := Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	defer stream.Close()

	first := stream.Result()
	second := stream.Result()
	requireAnswered(t, first)
	assert.Same(t, first, second, "Result returns the same latched message every time")
}

func TestEventStreamCloseMidStreamReportsAborted(t *testing.T) {
	t.Parallel()

	srv := newSlowStreamServer(t, 50, 20*time.Millisecond)
	defer srv.Close()

	stream := Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithMaxRetries(1),
	)

	// Read one event, then abandon the call.
	<-stream.Events()
	stream.Close()

	msg := stream.Result()
	require.NotNil(t, msg)
	assert.Equal(t, StopReasonAborted, msg.StopReason)
	assert.True(t, msg.StopReason.Terminal())
}

func TestEventStreamCloseIsSafeToCallTwice(t *testing.T) {
	t.Parallel()

	srv := newSlowStreamServer(t, 50, 20*time.Millisecond)
	defer srv.Close()

	stream := Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithMaxRetries(1),
	)
	stream.Close()
	stream.Close()

	assert.Equal(t, StopReasonAborted, stream.Result().StopReason)
}

func TestEventStreamCloseAfterCompletionKeepsTheAnswer(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	stream := Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	msg := drain(stream)
	requireAnswered(t, msg)

	// Closing a finished call must not rewrite its outcome.
	stream.Close()
	assert.Equal(t, StopReasonStop, stream.Result().StopReason)
	assert.Equal(t, "hello", stream.Result().Content)
}

// A concurrent reader and a concurrent Result must be race-free.
func TestEventStreamConcurrentReaderAndResult(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	stream := Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)

	var wg sync.WaitGroup
	var seen int
	var fromResult *AssistantMessage

	wg.Add(2)
	go func() {
		defer wg.Done()
		for event := range stream.Events() {
			// Read through the snapshot while the producer keeps accumulating.
			_ = event.Message.Content
			seen++
		}
	}()
	go func() {
		defer wg.Done()
		fromResult = stream.Result()
	}()
	wg.Wait()

	assert.Positive(t, seen)
	requireAnswered(t, fromResult)
	assert.Equal(t, "hello", fromResult.Content)
}

func TestEventStreamCancelledParentContextReportsAborted(t *testing.T) {
	t.Parallel()

	srv := newSlowStreamServer(t, 50, 20*time.Millisecond)
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	stream := Stream(ctx, RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithMaxRetries(1),
	)
	<-stream.Events()
	cancel()

	msg := stream.Result()
	assert.Equal(t, StopReasonAborted, msg.StopReason, "the caller's cancellation is an abort, not an error")
}

func TestEventStreamReportsFailedLegs(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			return
		}
		if model == testModelA {
			http.Error(w, "quota exhausted", http.StatusTooManyRequests)
			return
		}
		writeChatSuccess(t, w, "fallback answer", "stop")
	}))
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireAnswered(t, msg)

	failed := eventsOfKind(events, EventLegFailed)
	require.Len(t, failed, 1, "the chain announced the leg it gave up on")
	require.NotNil(t, failed[0].Attempt)
	assert.Equal(t, "openai/model-a", failed[0].Attempt.Model)
	require.NotNil(t, failed[0].Attempt.Err)
	assert.Equal(t, "fallback answer", msg.Content)
	require.Len(t, msg.Attempts, 2)
}

// A leg's partial text must not leak into the next leg's answer.
func TestFailedLegTextDoesNotLeakIntoTheNextLeg(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			return
		}
		if model == testModelA {
			// Text, then a truncation the guards reject.
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = fmt.Fprint(w,
				"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"partial thinking\"}}]}\n\n"+
					"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n"+
					"data: [DONE]\n\n")
			return
		}
		writeChatSuccess(t, w, "clean answer", "stop")
	}))
	defer srv.Close()

	msg := drain(Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,gemini/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireAnswered(t, msg)

	assert.Equal(t, "clean answer", msg.Content)
	assert.Empty(t, msg.Reasoning, "the failed leg's thinking is discarded")
	assert.Equal(t, "gemini/model-b", msg.Model)
}
