package smolllm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAskSurfacesFinishReasonVerbatim(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		finishReason string
		finalChoice  string
	}{
		{name: "standard", finishReason: "length", finalChoice: `{"delta":{},"finish_reason":"length"}`},
		{
			name:         "nonstandard",
			finishReason: "provider-specific",
			finalChoice:  `{"delta":{},"finish_reason":"provider-specific"}`,
		},
		{name: "omitted", finishReason: "", finalChoice: `{"delta":{}}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			srv := newChatStreamServer(t, tt.finalChoice)
			defer srv.Close()

			resp, err := Ask(context.Background(), PromptFromString("hi"),
				WithModel("openai/gpt-5"),
				WithBaseURL(srv.URL+"/"),
				WithAPIKey("test-key"),
			)
			require.NoError(t, err)
			assert.Equal(t, "hello", resp.Text)
			assert.Equal(t, tt.finishReason, resp.FinishReason)
		})
	}
}

func TestStreamSurfacesFinishReasonAfterCompletion(t *testing.T) {
	t.Parallel()

	srv := newChatStreamServer(t, `{"delta":{},"finish_reason":"content_filter"}`)
	defer srv.Close()

	resp, err := Stream(context.Background(), PromptFromString("hi"),
		WithModel("openai/gpt-5"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)

	var content strings.Builder
	for chunk := range resp.Stream.Chan() {
		content.WriteString(chunk.Content)
	}
	require.NoError(t, resp.Stream.Wait())
	assert.Equal(t, "hello", content.String())
	assert.Equal(t, "content_filter", resp.FinishReason)
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
		assert.NoError(t, err)
	}))
}
