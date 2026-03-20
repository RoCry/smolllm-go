package smolllm

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildRequestPayloadBasic(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	url, body, tokens, err := buildRequestPayload(prompt, "", "gpt-4o-mini", "openai", "https://api.openai.com", nil, nil, nil, nil)
	require.NoError(t, err)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", url)
	assert.Positive(t, tokens)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, "gpt-4o-mini", payload["model"])
	assert.Equal(t, true, payload["stream"])
}

func TestBuildRequestPayloadWithSamplingParams(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	temp := 0.4
	topP := 0.85
	_, body, _, err := buildRequestPayload(prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, &temp, &topP, nil)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, temp, payload["temperature"])
	assert.Equal(t, topP, payload["top_p"])
}

func TestComposeMessagesRejectsImageOnAssistant(t *testing.T) {
	t.Parallel()
	prompt := PromptFromMessages([]Message{Assistant("hi")})
	_, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.Error(t, err)
}

func TestComposeMessagesRejectsMultipleMessagesWithImages(t *testing.T) {
	t.Parallel()
	prompt := PromptFromMessages([]Message{User("one"), User("two")})
	_, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.Error(t, err)
}

func TestBuildRequestPayloadRequiresBaseURL(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	_, _, _, err := buildRequestPayload(prompt, "", "gpt-4o", "openai", "", nil, nil, nil, nil)
	require.Error(t, err)
}

func TestBuildRequestPayloadRejectsInvalidTemperature(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	temp := 3.0
	_, _, _, err := buildRequestPayload(prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, &temp, nil, nil)
	require.Error(t, err)
}

func TestBuildRequestPayloadRejectsInvalidTopP(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	topP := -0.1
	_, _, _, err := buildRequestPayload(prompt, "", "gpt-4o", "openai", "https://api.openai.com", nil, nil, &topP, nil)
	require.Error(t, err)
}

func TestBuildRequestPayloadWithReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hello")
	effort := "medium"
	_, body, _, err := buildRequestPayload(prompt, "", "o3", "openai", "https://api.openai.com", nil, nil, nil, &effort)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, "medium", payload["reasoning_effort"])
}

func TestBuildRequestPayloadRejectsInvalidReasoningEffort(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("hi")
	effort := "extreme"
	_, _, _, err := buildRequestPayload(prompt, "", "o3", "openai", "https://api.openai.com", nil, nil, nil, &effort)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "reasoning_effort")
}

func TestComposeMessagesAllowsSystemWithImagesAndSingleUser(t *testing.T) {
	t.Parallel()
	// This tests the fix for the CLI issue where --system with --images was rejected
	prompt := PromptFromMessages([]Message{System("analyze this"), User("what do you see?")})
	messages, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.NoError(t, err, "system + single user message with images should be allowed")
	assert.Len(t, messages, 2, "should have system and user messages")
}

func TestComposeMessagesRejectsMultipleUserMessagesWithImages(t *testing.T) {
	t.Parallel()
	// Even with system prompt, multiple user messages should still be rejected
	prompt := PromptFromMessages([]Message{System("sys"), User("one"), User("two")})
	_, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.Error(t, err, "multiple user messages with images should be rejected")
}
