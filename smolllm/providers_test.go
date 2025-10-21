package smolllm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestParseModelStringWithExplicitModel(t *testing.T) {
	t.Parallel()
	prov, model, err := parseModelString("openai/gpt-4o-mini")
	require.NoError(t, err)
	require.Equal(t, "openai", prov.Name)
	require.Equal(t, "gpt-4o-mini", model)
}

func TestParseModelStringUsesDefaultModel(t *testing.T) {
	t.Parallel()
	prov, model, err := parseModelString("gemini")
	require.NoError(t, err)
	require.Equal(t, "gemini", prov.Name)
	require.Equal(t, "gemini-2.0-flash", model)
}

func TestParseModelStringEnvOverrides(t *testing.T) {
	t.Setenv("CUSTOM_BASE_URL", "https://custom.example")
	prov, model, err := parseModelString("custom/model-x")
	require.NoError(t, err)
	require.Equal(t, "https://custom.example", prov.BaseURL)
	require.Equal(t, "model-x", model)
}

func TestParseModelStringErrors(t *testing.T) {
	_, _, err := parseModelString("")
	require.Error(t, err)

	t.Setenv("UNKNOWN_BASE_URL", "")
	_, _, err = parseModelString("unknown")
	require.Error(t, err)
}
