package smolllm

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testAPIKey  = "test-key"
	testBaseURL = "https://example.com"
)

func TestPrepareLLMCallKeepsModelNameOpaque(t *testing.T) {
	t.Parallel()

	// Everything after the first "/" is the wire model name, verbatim: the
	// library never interprets punctuation inside it.
	tests := []struct {
		name          string
		spec          string
		wantProvider  string
		wantModelName string
	}{
		{
			name:          "tilde prefix survives",
			spec:          "openrouter/~deepseek/x",
			wantProvider:  "openrouter",
			wantModelName: "~deepseek/x",
		},
		{
			name:          "nested slashes survive",
			spec:          "groq/qwen/qwen3-32b",
			wantProvider:  "groq",
			wantModelName: "qwen/qwen3-32b",
		},
		{
			name:          "bang is no longer a reasoning-effort separator",
			spec:          "openai/gpt-5!none",
			wantProvider:  "openai",
			wantModelName: "gpt-5!none",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			opts := defaultOptions()
			opts.DefaultProvider = testProviderConfig(testBaseURL, testAPIKey)

			call, err := prepareLLMCall(RequestFromString("hello"), opts, tt.spec, newBalancer())
			require.NoError(t, err)

			assert.Equal(t, tt.wantProvider, call.Provider.Name)
			assert.Equal(t, tt.spec, call.Model)
			assert.Equal(t, tt.wantModelName, call.ModelName)

			var payload map[string]any
			require.NoError(t, json.Unmarshal(call.Body, &payload))
			assert.Equal(t, tt.wantModelName, payload["model"])
		})
	}
}

func TestPrepareLLMCallBareModelURLGrammar(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		baseURL string
		wantURL string
	}{
		{
			name:    "plain base inserts v1",
			baseURL: "https://example.com",
			wantURL: "https://example.com/v1/chat/completions",
		},
		{
			name:    "trailing hash is verbatim",
			baseURL: "https://example.com/custom/endpoint#",
			wantURL: "https://example.com/custom/endpoint",
		},
		{
			name:    "trailing slash appends endpoint",
			baseURL: "https://example.com/api/",
			wantURL: "https://example.com/api/chat/completions",
		},
		{
			name:    "version suffix respected",
			baseURL: "https://example.com/v2",
			wantURL: "https://example.com/v2/chat/completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			opts := defaultOptions()
			opts.DefaultProvider = testProviderConfig(tt.baseURL, testAPIKey)

			call, err := prepareLLMCall(RequestFromString("hello"), opts, "gpt-4", newBalancer())
			require.NoError(t, err)

			assert.Equal(t, tt.wantURL, call.URL)
			assert.Empty(t, call.Provider.Name)
			assert.Equal(t, "gpt-4", call.Model)
			assert.Equal(t, "gpt-4", call.ModelName)
		})
	}
}

func TestPrepareLLMCallBareModelRequiresExplicitBaseURL(t *testing.T) {
	// Bare mode must never derive env keys from the empty provider name.
	t.Setenv("_BASE_URL", "https://env.example")

	opts := defaultOptions()
	opts.DefaultProvider = testProviderConfig("", testAPIKey)

	_, err := prepareLLMCall(RequestFromString("hello"), opts, "gpt-4", newBalancer())
	require.EqualError(t, err,
		`bare model "gpt-4" requires a base URL. Provide WithProvider(BareProvider, ...) or use provider/model format`)
}

func TestPrepareLLMCallBareModelRequiresExplicitAPIKey(t *testing.T) {
	// Bare mode must never derive env keys from the empty provider name.
	t.Setenv("_API_KEY", "env-key")

	opts := defaultOptions()
	opts.DefaultProvider = testProviderConfig(testBaseURL, "")

	_, err := prepareLLMCall(RequestFromString("hello"), opts, "gpt-4", newBalancer())
	require.EqualError(t, err,
		`bare model "gpt-4" requires an API key. Provide WithProvider(BareProvider, ...) or use provider/model format`)
}
