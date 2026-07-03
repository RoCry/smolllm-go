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

func stringPtr(value string) *string {
	return &value
}
