package smolllm

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestOptionsBuilders(t *testing.T) {
	t.Parallel()
	handlerCalled := false
	handler := func(context.Context, string) error {
		handlerCalled = true
		return nil
	}

	client := &http.Client{}

	optFns := []Option{
		WithSystemPrompt("be terse"),
		WithModel("openai/gpt-4o,gemini/gemini-2.0-flash"),
		WithAPIKey("k1,k2"),
		WithBaseURL("https://example.com"),
		WithImagePaths("img1", "img2"),
		WithTimeout(5 * time.Second),
		WithBacktickRemoval(),
		WithStreamHandler(handler),
		WithHTTPClient(client),
	}

	opts := defaultOptions()
	for _, fn := range optFns {
		fn(&opts)
	}

	require.Equal(t, "be terse", opts.SystemPrompt)
	require.Equal(t, "openai/gpt-4o,gemini/gemini-2.0-flash", opts.Model)
	require.Equal(t, "k1,k2", opts.APIKey)
	require.Equal(t, "https://example.com", opts.BaseURL)
	require.Equal(t, []string{"img1", "img2"}, opts.ImagePaths)
	require.Equal(t, 5*time.Second, opts.Timeout)
	require.True(t, opts.RemoveBackticks)
	require.NotNil(t, opts.StreamHandler)
	require.Equal(t, client, opts.HTTPClient)

	require.NoError(t, opts.StreamHandler(context.Background(), "delta"))
	require.True(t, handlerCalled)
	require.Equal(t, "img1", opts.ImagePaths[0])
}
