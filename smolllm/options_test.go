package smolllm

import (
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOptionsBuilders(t *testing.T) {
	t.Parallel()
	client := new(http.Client)
	logger := slog.New(slog.DiscardHandler)

	optFns := []Option{
		WithModel("openai/gpt-4o,gemini/gemini-2.0-flash"),
		WithTemperature(0.7),
		WithTopP(0.9),
		WithMaxTokens(128),
		WithStop("END", "STOP"),
		WithSeed(42),
		WithReasoningEffort("medium"),
		withTestProvider("https://example.com", "k1,k2"),
		WithImagePaths("img1", "img2"),
		WithTimeout(5 * time.Second),
		WithBacktickRemoval(),
		WithHTTPClient(client),
		WithLogger(logger),
	}

	opts := defaultOptions()
	for _, fn := range optFns {
		fn(&opts)
	}

	assert.Equal(t, "openai/gpt-4o,gemini/gemini-2.0-flash", opts.Model)
	require.NotNil(t, opts.Temperature)
	require.NotNil(t, opts.TopP)
	require.NotNil(t, opts.MaxTokens)
	require.NotNil(t, opts.Seed)
	assert.InDelta(t, 0.7, *opts.Temperature, 1e-9)
	assert.InDelta(t, 0.9, *opts.TopP, 1e-9)
	assert.Equal(t, 128, *opts.MaxTokens)
	assert.Equal(t, []string{"END", "STOP"}, opts.Stop)
	assert.Equal(t, 42, *opts.Seed)
	require.NotNil(t, opts.ReasoningEffort)
	assert.Equal(t, "medium", *opts.ReasoningEffort)
	assert.Equal(t, "k1,k2", opts.DefaultProvider.APIKey)
	assert.Equal(t, "https://example.com", opts.DefaultProvider.BaseURL)
	assert.Equal(t, []string{"img1", "img2"}, opts.ImagePaths)
	assert.Equal(t, 5*time.Second, opts.Timeout)
	assert.True(t, opts.RemoveBackticks)
	assert.Equal(t, client, opts.HTTPClient)
	assert.Equal(t, logger, opts.Logger)

	assert.Equal(t, "img1", opts.ImagePaths[0])
}

func TestWithProviderResolvesPerProvider(t *testing.T) {
	t.Parallel()

	opts := applyOptions(
		WithDefaultProvider(testProviderConfig("https://default.example", "default-key")),
		WithProvider("openai", testProviderConfig("https://openai.example", "")),
		WithProvider(BareProvider, testProviderConfig("", "bare-key")),
	)

	// A field the entry leaves empty falls through to the default rather than
	// masking it.
	assert.Equal(t, "https://openai.example", opts.providerBaseURL("openai"))
	assert.Equal(t, "default-key", opts.providerAPIKey("openai"))
	assert.Equal(t, "https://default.example", opts.providerBaseURL(BareProvider))
	assert.Equal(t, "bare-key", opts.providerAPIKey(BareProvider))
	assert.Equal(t, "https://default.example", opts.providerBaseURL("groq"))
}

func TestWithProviderDoesNotMutateBaseOptions(t *testing.T) {
	t.Parallel()

	// A per-call option must never write into the map a Client was built with.
	base := applyOptions(WithProvider("openai", testProviderConfig("https://base.example", "base-key")))
	perCall := base
	WithProvider("openai", testProviderConfig("https://call.example", "call-key"))(&perCall)

	assert.Equal(t, "https://base.example", base.providerBaseURL("openai"))
	assert.Equal(t, "https://call.example", perCall.providerBaseURL("openai"))
}

func TestWithProviderDoesNotAliasCallerHeaders(t *testing.T) {
	t.Parallel()

	headers := map[string]string{"X-Tenant": "original"}
	opts := applyOptions(WithProvider("openai", ProviderConfig{BaseURL: "", APIKey: "", Headers: headers}))
	headers["X-Tenant"] = testMutated

	assert.Equal(t, "original", opts.providerHeaders("openai")["X-Tenant"])
}

func TestWithMaxRetries(t *testing.T) {
	t.Parallel()

	opts := applyOptions()
	assert.Equal(t, defaultMaxRetries, opts.MaxRetries, "retrying is on by default")

	WithMaxRetries(1)(&opts)
	assert.Equal(t, 1, opts.MaxRetries, "one attempt disables retrying")
}

func TestWithMaxRetriesPanicsOnNonPositive(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithMaxRetries: attempts must be positive", func() {
		WithMaxRetries(0)
	})
	require.PanicsWithValue(t, "WithMaxRetries: attempts must be positive", func() {
		WithMaxRetries(-1)
	})
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

func TestWithMaxTokensPanicsOnInvalid(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithMaxTokens: value must be positive", func() {
		WithMaxTokens(0)
	})
}

func TestWithStopPanicsOnEmpty(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithStop: at least one stop sequence required", func() {
		WithStop()
	})
	require.PanicsWithValue(t, "WithStop: stop sequences must not be empty", func() {
		WithStop("END", " ")
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

func TestWithDimensions(t *testing.T) {
	t.Parallel()
	opts := defaultOptions()
	WithDimensions(512)(&opts)
	assert.Equal(t, 512, opts.Dimensions)
}

func TestWithDimensionsPanicsOnNonPositive(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "WithDimensions: dimensions must be positive", func() {
		WithDimensions(0)
	})
	require.PanicsWithValue(t, "WithDimensions: dimensions must be positive", func() {
		WithDimensions(-1)
	})
}
