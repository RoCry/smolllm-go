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
		withTestProvider("", "sk-test-primary,sk-test-secondary"),
	)
	require.NoError(t, err)
}

func TestValidateBareModel(t *testing.T) {
	t.Parallel()

	t.Run("succeeds with explicit base URL and API key", func(t *testing.T) {
		t.Parallel()

		err := Validate(
			WithModel("gpt-4"),
			withTestProvider("https://bare.example", "test-key"),
		)
		require.NoError(t, err)
	})

	t.Run("fails without base URL", func(t *testing.T) {
		t.Parallel()

		err := Validate(
			WithModel("gpt-4"),
			withTestProvider("", "test-key"),
		)
		require.Error(t, err)
		assert.ErrorContains(t, err,
			`bare model "gpt-4" requires a base URL. Provide WithProvider(BareProvider, ...) or use provider/model format`)
	})

	t.Run("fails without API key", func(t *testing.T) {
		t.Parallel()

		err := Validate(
			WithModel("gpt-4"),
			withTestProvider("https://bare.example", ""),
		)
		require.Error(t, err)
		assert.ErrorContains(t, err,
			`bare model "gpt-4" requires an API key. Provide WithProvider(BareProvider, ...) or use provider/model format`)
	})
}

func TestValidateRejectsUnsupportedGlobalReasoningEffort(t *testing.T) {
	t.Parallel()

	err := Validate(
		WithModel("openai/gpt-5"),
		withTestProvider("", "test-key"),
		WithReasoningEffort("minimum"),
	)
	require.Error(t, err)
	assert.ErrorContains(t, err, "reasoning_effort")
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
		withTestProvider("https://api.openai.com/v1,https://alt.example/v1,https://third.example/v1", "sk-one,sk-two"),
	)
	require.Error(t, err)
	assert.ErrorContains(t, err, "counts must match")
}

func TestValidateAcceptsExplicitBaseURLForUnknownProvider(t *testing.T) {
	t.Setenv("CUSTOM_BASE_URL", "")

	err := Validate(
		WithModel("custom/model-x"),
		withTestProvider("https://custom.example/v1", "test-key"),
	)
	require.NoError(t, err)
}

func TestValidateUnknownProviderNamesBaseURLRemedies(t *testing.T) {
	t.Setenv("CUSTOM_BASE_URL", "")

	err := Validate(
		WithModel("custom/model-x"),
		withTestProvider("", "test-key"),
	)
	require.Error(t, err)
	require.ErrorContains(t, err, "CUSTOM_BASE_URL")
	require.ErrorContains(t, err, "WithProvider")
}
