package smolllm

import (
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/require"
)

func TestPromptFromStringSetsRoleAndContent(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("ping")
	require.Len(t, prompt.Messages, 1)
	role, ok := messageRole(prompt.Messages[0])
	require.True(t, ok)
	require.Equal(t, "user", role)
	require.NotNil(t, prompt.Messages[0].GetContent().AsAny())
}

func TestComposeMessagesWithSystem(t *testing.T) {
	t.Parallel()
	prompt := PromptFromMessages([]Message{User("hello")})
	msgs, err := composeMessages(prompt, "act concise", nil)
	require.NoError(t, err)
	require.Len(t, msgs, 2)
	require.Equal(t, "system", *msgs[0].GetRole())
	require.Equal(t, "user", *msgs[1].GetRole())
	require.NotNil(t, msgs[1].GetContent().AsAny())
}

func TestComposeMessagesWithImages(t *testing.T) {
	t.Parallel()
	prompt := PromptFromString("describe photo")
	msgs, err := composeMessages(prompt, "", []string{"data:image/png;base64,AA=="})
	require.NoError(t, err)
	require.Len(t, msgs, 1)
	partsAny := msgs[0].GetContent().AsAny()
	partsPtr, ok := partsAny.(*[]openai.ChatCompletionContentPartUnionParam)
	require.True(t, ok)
	require.Len(t, *partsPtr, 2)
}
