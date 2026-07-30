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

func TestPrepareLLMCallUsesReasoningEffortSuffix(t *testing.T) {
	t.Parallel()

	opts := defaultOptions()
	opts.APIKey = testAPIKey
	opts.BaseURL = testBaseURL

	call, err := prepareLLMCall(PromptFromString("hello"), opts, "groq/qwen/qwen3-32b!none")
	require.NoError(t, err)

	assert.Equal(t, "groq", call.Provider.Name)
	assert.Equal(t, "groq/qwen/qwen3-32b", call.Model)
	assert.Equal(t, "qwen/qwen3-32b", call.ModelName)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(call.Body, &payload))
	assert.Equal(t, "qwen/qwen3-32b", payload["model"])
	assert.Equal(t, "none", payload["reasoning_effort"])
}

func TestPrepareLLMCallReasoningEffortSuffixOverridesOption(t *testing.T) {
	t.Parallel()

	opts := defaultOptions()
	opts.APIKey = testAPIKey
	opts.BaseURL = testBaseURL
	opts.ReasoningEffort = stringPtr("medium")

	call, err := prepareLLMCall(PromptFromString("hello"), opts, "openai/gpt-5!high")
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(call.Body, &payload))
	assert.Equal(t, "gpt-5", payload["model"])
	assert.Equal(t, "high", payload["reasoning_effort"])
}

func TestPrepareLLMCallIgnoresEmptyReasoningEffortSuffix(t *testing.T) {
	t.Parallel()

	opts := defaultOptions()
	opts.APIKey = testAPIKey
	opts.BaseURL = testBaseURL
	opts.ReasoningEffort = stringPtr("medium")

	call, err := prepareLLMCall(PromptFromString("hello"), opts, "openai/gpt-5!")
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(call.Body, &payload))
	assert.Equal(t, "gpt-5", payload["model"])
	assert.Equal(t, "medium", payload["reasoning_effort"])
}

func TestPrepareLLMCallTrimsReasoningEffortSuffix(t *testing.T) {
	t.Parallel()

	opts := defaultOptions()
	opts.APIKey = testAPIKey
	opts.BaseURL = testBaseURL

	call, err := prepareLLMCall(PromptFromString("hello"), opts, "  openai/gpt-5  ! low ")
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(call.Body, &payload))
	assert.Equal(t, "gpt-5", payload["model"])
	assert.Equal(t, "low", payload["reasoning_effort"])
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
			opts.APIKey = testAPIKey
			opts.BaseURL = tt.baseURL

			call, err := prepareLLMCall(PromptFromString("hello"), opts, "gpt-4")
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
	opts.APIKey = testAPIKey

	_, err := prepareLLMCall(PromptFromString("hello"), opts, "gpt-4")
	require.EqualError(t, err, `bare model "gpt-4" requires a base URL. Provide WithBaseURL or use provider/model format`)
}

func TestPrepareLLMCallBareModelRequiresExplicitAPIKey(t *testing.T) {
	// Bare mode must never derive env keys from the empty provider name.
	t.Setenv("_API_KEY", "env-key")

	opts := defaultOptions()
	opts.BaseURL = testBaseURL

	_, err := prepareLLMCall(PromptFromString("hello"), opts, "gpt-4")
	require.EqualError(t, err, `bare model "gpt-4" requires an API key. Provide WithAPIKey or use provider/model format`)
}

func stringPtr(value string) *string {
	return &value
}
