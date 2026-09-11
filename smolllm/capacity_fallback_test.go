package smolllm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStreamAdvancesPastProviderCapacityLimit(t *testing.T) {
	t.Parallel()

	const capacityError = `{"error":{"message":"Request too large: TPM limit 8000, requested 8732",` +
		`"type":"tokens","code":"rate_limit_exceeded"}}`
	var firstAttempts, fallbackAttempts atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model, err := requestModel(r)
		if err != nil {
			t.Errorf("decode fake provider request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		switch model {
		case testModelA:
			firstAttempts.Add(1)
			http.Error(w, capacityError, http.StatusRequestEntityTooLarge)
		case testModelB:
			fallbackAttempts.Add(1)
			writeChatSuccess(t, w, "fallback answer", "stop")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("hi"),
		WithModel("groq/model-a,openai/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireAnswered(t, msg)
	assert.Equal(t, "fallback answer", deltaText(events, EventTextDelta))
	assert.Equal(t, "fallback answer", msg.Content)
	assert.Empty(t, msg.ErrorMessage)
	assert.Equal(t, int32(1), firstAttempts.Load(), "capacity failure must not retry the same provider")
	assert.Equal(t, int32(1), fallbackAttempts.Load())
	require.Len(t, msg.Attempts, 2)
	assert.Equal(t, "groq/model-a", msg.Attempts[0].Model)
	require.NotNil(t, msg.Attempts[0].Err)
	assert.Equal(t, http.StatusRequestEntityTooLarge, msg.Attempts[0].Err.StatusCode)
	assert.Equal(t, DispositionAdvance, msg.Attempts[0].Err.Disposition)
	assert.Contains(t, msg.Attempts[0].Err.Error(), capacityError)
	assert.Equal(t, "openai/model-b", msg.Attempts[1].Model)
	assert.Nil(t, msg.Attempts[1].Err)
	require.NotEmpty(t, events)
	assert.Equal(t, EventDone, events[len(events)-1].Kind)
	for _, event := range events[:len(events)-1] {
		assert.False(t, event.Kind.Terminal(), "failed first provider must not terminate the stream")
	}
}
