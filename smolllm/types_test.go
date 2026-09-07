package smolllm

import (
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRequestFromStringSetsRoleAndContent(t *testing.T) {
	t.Parallel()
	req := RequestFromString("ping")
	require.Len(t, req.Messages, 1)
	role, ok := messageRole(req.Messages[0])
	require.True(t, ok)
	require.Equal(t, "user", role)
	require.NotNil(t, req.Messages[0].GetContent().AsAny())
}

func TestComposeMessagesWithSystem(t *testing.T) {
	t.Parallel()
	req := RequestFromMessages([]Message{User("hello")})
	req.System = "act concise"
	msgs, err := composeMessages(req, nil)
	require.NoError(t, err)
	assert.Len(t, msgs, 2)
	assert.Equal(t, "system", *msgs[0].GetRole())
	assert.Equal(t, "user", *msgs[1].GetRole())
	assert.NotNil(t, msgs[1].GetContent().AsAny())
}

func TestComposeMessagesWithImages(t *testing.T) {
	t.Parallel()
	req := RequestFromString("describe photo")
	msgs, err := composeMessages(req, []string{testImageDataURL})
	require.NoError(t, err)
	assert.Len(t, msgs, 1)
	partsAny := msgs[0].GetContent().AsAny()
	partsPtr, ok := partsAny.(*[]openai.ChatCompletionContentPartUnionParam)
	require.True(t, ok)
	assert.Len(t, *partsPtr, 2)
}
