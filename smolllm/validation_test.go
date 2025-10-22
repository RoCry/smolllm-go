package smolllm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestValidateSucceedsWithExplicitOptions(t *testing.T) {
	t.Parallel()

	err := Validate(
		WithModel("openai/gpt-4o-mini"),
		WithAPIKey("sk-test-primary,sk-test-secondary"),
	)
	require.NoError(t, err)
}

func TestValidateRequiresModel(t *testing.T) {
	t.Parallel()

	err := Validate()
	require.Error(t, err)
	assert.ErrorContains(t, err, "model string not provided")
}

func TestValidateDetectsKeyURLMismatch(t *testing.T) {
	t.Parallel()

	err := Validate(
		WithModel("openai/gpt-4o-mini"),
		WithAPIKey("sk-one,sk-two"),
		WithBaseURL("https://api.openai.com/v1,https://alt.example/v1,https://third.example/v1"),
	)
	require.Error(t, err)
	assert.ErrorContains(t, err, "counts must match")
}
