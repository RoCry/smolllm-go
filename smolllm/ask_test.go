package smolllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
