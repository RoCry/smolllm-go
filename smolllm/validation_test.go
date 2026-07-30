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

func TestValidateBareModel(t *testing.T) {
	t.Parallel()

	t.Run("succeeds with explicit base URL and API key", func(t *testing.T) {
		t.Parallel()

		err := Validate(
			WithModel("gpt-4!low"),
			WithAPIKey("test-key"),
			WithBaseURL("https://bare.example"),
		)
		require.NoError(t, err)
	})

	t.Run("fails without base URL", func(t *testing.T) {
		t.Parallel()

		err := Validate(
			WithModel("gpt-4"),
			WithAPIKey("test-key"),
		)
		require.Error(t, err)
		assert.ErrorContains(t, err,
			`bare model "gpt-4" requires a base URL. Provide WithBaseURL or use provider/model format`)
	})

	t.Run("fails without API key", func(t *testing.T) {
		t.Parallel()

		err := Validate(
			WithModel("gpt-4"),
			WithBaseURL("https://bare.example"),
		)
		require.Error(t, err)
		assert.ErrorContains(t, err,
			`bare model "gpt-4" requires an API key. Provide WithAPIKey or use provider/model format`)
	})
}

func TestValidateRejectsUnsupportedReasoningEffortSuffix(t *testing.T) {
	t.Parallel()

	err := Validate(
		WithModel("openai/gpt-5!minimum"),
		WithAPIKey("test-key"),
	)
	require.Error(t, err)
	assert.ErrorContains(t, err, "reasoning_effort")
}

func TestValidateRejectsUnsupportedGlobalReasoningEffort(t *testing.T) {
	t.Parallel()

	err := Validate(
		WithModel("openai/gpt-5"),
		WithAPIKey("test-key"),
		WithReasoningEffort("minimum"),
	)
	require.Error(t, err)
	assert.ErrorContains(t, err, "reasoning_effort")
}

func TestValidateSuffixOverridesUnsupportedGlobalReasoningEffort(t *testing.T) {
	t.Parallel()

	err := Validate(
		WithModel("ollama/qwen!none"),
		WithReasoningEffort("minimal"),
	)
	require.NoError(t, err)
}

func TestValidateRequiresModel(t *testing.T) {
	t.Setenv("SMOLLLM_MODEL", "")

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

func TestValidateAcceptsExplicitBaseURLForUnknownProvider(t *testing.T) {
	t.Setenv("CUSTOM_BASE_URL", "")

	err := Validate(
		WithModel("custom/model-x"),
		WithAPIKey("test-key"),
		WithBaseURL("https://custom.example/v1"),
	)
	require.NoError(t, err)
}

func TestValidateUnknownProviderNamesBaseURLRemedies(t *testing.T) {
	t.Setenv("CUSTOM_BASE_URL", "")

	err := Validate(
		WithModel("custom/model-x"),
		WithAPIKey("test-key"),
	)
	require.Error(t, err)
	require.ErrorContains(t, err, "CUSTOM_BASE_URL")
	require.ErrorContains(t, err, "WithBaseURL")
}
