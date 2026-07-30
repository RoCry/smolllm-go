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

func TestParseModelStringBareModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		spec      string
		wantModel string
	}{
		{name: "plain bare model", spec: "gpt-4", wantModel: "gpt-4"},
		{name: "known provider name is just a model", spec: "gemini", wantModel: "gemini"},
		{name: "ollama name is just a model", spec: "ollama", wantModel: "ollama"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			prov, model, err := parseModelString(tt.spec)

			require.NoError(t, err)
			assert.Empty(t, prov.Name)
			assert.Empty(t, prov.BaseURL)
			assert.Equal(t, tt.wantModel, model)
		})
	}
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
			name:       "no suffix",
			spec:       "groq/qwen/qwen3-32b",
			wantModel:  "groq/qwen/qwen3-32b",
			wantEffort: nil,
		},
		{
			name:       "with effort",
			spec:       "groq/qwen/qwen3-32b!none",
			wantModel:  "groq/qwen/qwen3-32b",
			wantEffort: stringPtr("none"),
		},
		{
			name:       "bare model with effort",
			spec:       "gemini!low",
			wantModel:  "gemini",
			wantEffort: stringPtr("low"),
		},
		{
			name:       "trailing separator no value",
			spec:       "openai/gpt-5!",
			wantModel:  "openai/gpt-5",
			wantEffort: nil,
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

func TestParseModelStringErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		spec    string
		wantErr string
	}{
		{name: "empty string", spec: "", wantErr: "model string must not be empty"},
		{name: "empty provider", spec: "/gpt-4", wantErr: `provider name missing in model string "/gpt-4"`},
		{name: "empty model", spec: "openai/", wantErr: `model name missing for provider "openai"`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, _, err := parseModelString(tt.spec)

			require.EqualError(t, err, tt.wantErr)
		})
	}
}
