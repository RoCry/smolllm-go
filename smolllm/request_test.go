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
	url, body, tokens, err := buildRequestPayload(prompt, "", "gpt-4o-mini", "openai", "https://api.openai.com", nil)
	require.NoError(t, err)
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", url)
	assert.Positive(t, tokens)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, "gpt-4o-mini", payload["model"])
	assert.Equal(t, true, payload["stream"])
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
	_, _, _, err := buildRequestPayload(prompt, "", "gpt-4o", "openai", "", nil)
	require.Error(t, err)
}
