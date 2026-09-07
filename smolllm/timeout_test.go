package smolllm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// The whole-call deadline must bound the chain end to end. Three legs, each
// retried with 1s/2s backoff, would run for roughly nine seconds if the deadline
// only applied per attempt.
func TestTimeoutBoundsEveryLegAndItsBackoff(t *testing.T) {
	t.Parallel()

	var attempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempts.Add(1)
		http.Error(w, "upstream unavailable", http.StatusInternalServerError)
	}))
	defer srv.Close()

	start := time.Now()
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/a,openai/b,openai/c"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithTimeout(600*time.Millisecond),
	)
	elapsed := time.Since(start)

	requireFailed(t, msg)
	assert.Less(t, elapsed, 4*time.Second, "the deadline must cut the chain short, not bound each attempt")
	// The first backoff wait alone outlasts the deadline, so the chain cannot
	// reach every leg's full retry budget.
	assert.Less(t, int(attempts.Load()), 9)
}

func TestTimeoutCoversStreamConsumption(t *testing.T) {
	t.Parallel()

	// The provider sends one chunk and then holds the connection open, which is
	// what a stalled upstream looks like to a consumer.
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	stream := Stream(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithTimeout(400*time.Millisecond),
	)

	start := time.Now()
	events, msg := collect(stream)
	elapsed := time.Since(start)

	assert.Equal(t, "hello", deltaText(events, EventTextDelta), "the one chunk the provider managed to send")
	requireFailed(t, msg)
	assert.Contains(t, msg.ErrorMessage, "context deadline exceeded")
	assert.Less(t, elapsed, 5*time.Second)
}

func TestTimeoutZeroDisablesTheBound(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeChatSuccess(t, w, "hello", "stop")
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
		WithTimeout(0),
	)
	requireAnswered(t, msg)
	assert.Equal(t, "hello", msg.Content)
}
