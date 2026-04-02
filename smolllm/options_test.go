package smolllm

import (
	"context"
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOptionsBuilders(t *testing.T) {
	t.Parallel()
	handlerCalled := false
	handler := func(context.Context, string) error {
		handlerCalled = true
		return nil
	}

	client := new(http.Client)
	logger := slog.New(slog.DiscardHandler)

	optFns := []Option{
		WithSystemPrompt("be terse"),
		WithModel("openai/gpt-4o,gemini/gemini-2.0-flash"),
		WithTemperature(0.7),
		WithTopP(0.9),
		WithReasoningEffort("medium"),
		WithAPIKey("k1,k2"),
		WithBaseURL("https://example.com"),
		WithImagePaths("img1", "img2"),
		WithTimeout(5 * time.Second),
		WithBacktickRemoval(),
		WithStreamHandler(handler),
		WithHTTPClient(client),
		WithLogger(logger),
	}

	opts := defaultOptions()
	for _, fn := range optFns {
		fn(&opts)
	}

	assert.Equal(t, "be terse", opts.SystemPrompt)
	assert.Equal(t, "openai/gpt-4o,gemini/gemini-2.0-flash", opts.Model)
	require.NotNil(t, opts.Temperature)
	require.NotNil(t, opts.TopP)
	assert.Equal(t, 0.7, *opts.Temperature)
	assert.Equal(t, 0.9, *opts.TopP)
	require.NotNil(t, opts.ReasoningEffort)
	assert.Equal(t, "medium", *opts.ReasoningEffort)
	assert.Equal(t, "k1,k2", opts.APIKey)
	assert.Equal(t, "https://example.com", opts.BaseURL)
	assert.Equal(t, []string{"img1", "img2"}, opts.ImagePaths)
	assert.Equal(t, 5*time.Second, opts.Timeout)
	assert.True(t, opts.RemoveBackticks)
	assert.NotNil(t, opts.StreamHandler)
	assert.Equal(t, client, opts.HTTPClient)
	assert.Equal(t, logger, opts.Logger)

	require.NoError(t, opts.StreamHandler(context.Background(), "delta"))
	assert.True(t, handlerCalled)
	assert.Equal(t, "img1", opts.ImagePaths[0])
}

func TestWithLoggerPanicsOnNil(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithLogger: logger must not be nil", func() {
		WithLogger(nil)
	})
}

func TestWithTemperaturePanicsOnInvalid(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithTemperature: value must be between 0 and 2 inclusive", func() {
		WithTemperature(3)
	})
}

func TestWithTopPPanicsOnInvalid(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithTopP: value must be between 0 and 1 inclusive", func() {
		WithTopP(-0.1)
	})
}

func TestWithReasoningEffortPanicsOnEmpty(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithReasoningEffort: value must not be empty", func() {
		WithReasoningEffort("")
	})
}

func TestWithReasoningEffortAcceptsProviderSpecificValues(t *testing.T) {
	t.Parallel()
	for _, v := range []string{"none", "minimum", "low", "medium", "high", "xhigh"} {
		opts := defaultOptions()
		WithReasoningEffort(v)(&opts)
		require.NotNil(t, opts.ReasoningEffort)
		assert.Equal(t, v, *opts.ReasoningEffort)
	}
}
