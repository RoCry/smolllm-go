package smolllm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

// FinishReason is the provider's own word, kept verbatim. StopReason is the
// normalized answer derived from it.
func TestAskSurfacesFinishReasonVerbatimAndNormalizesStopReason(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		finishReason   string
		finalChoice    string
		wantStopReason StopReason
	}{
		{
			name:           "length truncation",
			finishReason:   "length",
			finalChoice:    `{"delta":{},"finish_reason":"length"}`,
			wantStopReason: StopReasonLength,
		},
		{
			name:           "content filter normalizes to stop",
			finishReason:   "content_filter",
			finalChoice:    `{"delta":{},"finish_reason":"content_filter"}`,
			wantStopReason: StopReasonStop,
		},
		{
			name:           "unknown provider string normalizes to stop",
			finishReason:   "provider-specific",
			finalChoice:    `{"delta":{},"finish_reason":"provider-specific"}`,
			wantStopReason: StopReasonStop,
		},
		{
			name:           "omitted",
			finishReason:   "",
			finalChoice:    `{"delta":{}}`,
			wantStopReason: StopReasonStop,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			srv := newChatStreamServer(t, tt.finalChoice)
			defer srv.Close()

			msg := Ask(context.Background(), RequestFromString("hi"),
				WithModel(testChatModel),
				withTestProvider(srv.URL+"/", "test-key"),
			)
			requireAnswered(t, msg)
			assert.Equal(t, "hello", msg.Content)
			assert.Equal(t, tt.finishReason, msg.FinishReason, "the provider string is never rewritten")
			assert.Equal(t, tt.wantStopReason, msg.StopReason)
		})
	}
}

func TestStreamSurfacesFinishReasonAfterCompletion(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"content_filter"}`)
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("hi"),
		WithModel(testChatModel),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireAnswered(t, msg)

	assert.Equal(t, "hello", deltaText(events, EventTextDelta))
	assert.Equal(t, "hello", msg.Content)
	assert.Equal(t, "content_filter", msg.FinishReason)
	assert.Equal(t, StopReasonStop, msg.StopReason)
}

func TestStreamExplicitBaseURLRescuesUnknownProvider(t *testing.T) {
	t.Setenv("CUSTOM_BASE_URL", "")

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	msg := drain(Stream(context.Background(), RequestFromString("hi"),
		WithModel("custom/model-x"),
		withTestProvider(srv.URL+"/", "test-key"),
	))
	requireAnswered(t, msg)

	assert.Equal(t, "hello", msg.Content)
	assert.Equal(t, "custom", msg.Provider)
	assert.Equal(t, "custom/model-x", msg.Model)
}

func TestStreamBareModelResolvesExplicitOptions(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"stop"}`)
	defer srv.Close()

	msg := drain(Stream(context.Background(), RequestFromString("hi"),
		WithModel("bare-model"),
		WithProvider(BareProvider, testProviderConfig(srv.URL+"/", "test-key")),
	))
	requireAnswered(t, msg)

	assert.Equal(t, "hello", msg.Content)
	assert.Empty(t, msg.Provider)
	assert.Equal(t, "bare-model", msg.Model)
	assert.Equal(t, "bare-model", msg.ModelName)
}

func TestStopReasonTerminal(t *testing.T) {
	t.Parallel()

	assert.False(t, StopReasonPending.Terminal(), "a partial snapshot is not terminal")
	for _, reason := range []StopReason{
		StopReasonStop, StopReasonLength, StopReasonToolUse, StopReasonError, StopReasonAborted,
	} {
		assert.True(t, reason.Terminal(), "%s ends a stream", reason)
	}
}

func newChatStreamServer(t *testing.T, finalChoice string) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := fmt.Fprintf(w,
			"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"+
				"data: {\"choices\":[%s]}\n\n"+
				"data: [DONE]\n\n",
			finalChoice,
		)
		if err != nil {
			t.Errorf("write fake provider response: %v", err)
			return
		}
	}))
}
