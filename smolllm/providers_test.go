package smolllm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseModelStringWithExplicitModel(t *testing.T) {
	t.Parallel()
	prov, model, err := parseModelString("openai/gpt-4o-mini")
	require.NoError(t, err)
	assert.Equal(t, "openai", prov.Name)
	assert.Equal(t, "gpt-4o-mini", model)
}

func TestParseModelStringUsesDefaultModel(t *testing.T) {
	t.Parallel()
	prov, model, err := parseModelString("gemini")
	require.NoError(t, err)
	assert.Equal(t, "gemini", prov.Name)
	assert.Equal(t, "gemini-2.0-flash", model)
}

func TestParseModelSpec(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		spec       string
		wantModel  string
		wantEffort *string
	}{
		{
			name:      "no suffix",
			spec:      "groq/qwen/qwen3-32b",
			wantModel: "groq/qwen/qwen3-32b",
		},
		{
			name:       "with effort",
			spec:       "groq/qwen/qwen3-32b!none",
			wantModel:  "groq/qwen/qwen3-32b",
			wantEffort: stringPtr("none"),
		},
		{
			name:       "provider default with effort",
			spec:       "gemini!low",
			wantModel:  "gemini",
			wantEffort: stringPtr("low"),
		},
		{
			name:      "trailing separator no value",
			spec:      "openai/gpt-5!",
			wantModel: "openai/gpt-5",
		},
		{
			name:       "strips whitespace",
			spec:       "  openai/gpt-5  ! medium ",
			wantModel:  "openai/gpt-5",
			wantEffort: stringPtr("medium"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, effort := parseModelSpec(tt.spec)

			assert.Equal(t, tt.wantModel, model)
			assert.Equal(t, tt.wantEffort, effort)
		})
	}
}

func TestParseModelStringEnvOverrides(t *testing.T) {
	t.Setenv("CUSTOM_BASE_URL", "https://custom.example")
	prov, model, err := parseModelString("custom/model-x")
	require.NoError(t, err)
	assert.Equal(t, "https://custom.example", prov.BaseURL)
	assert.Equal(t, "model-x", model)
}

func TestParseModelStringErrors(t *testing.T) {
	_, _, err := parseModelString("")
	require.Error(t, err)

	t.Setenv("UNKNOWN_BASE_URL", "")
	_, _, err = parseModelString("unknown")
	require.Error(t, err)
}
