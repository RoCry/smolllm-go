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
	assert.Equal(t, providerOpenAI, prov.Name)
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
		{name: "known provider name is just a model", spec: providerGemini, wantModel: providerGemini},
		{name: "ollama name is just a model", spec: providerOllama, wantModel: providerOllama},
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

// Everything after the first "/" is an opaque model name and reaches the wire
// verbatim, punctuation included.
func TestParseModelStringKeepsModelNameOpaque(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		spec         string
		wantProvider string
		wantModel    string
	}{
		{name: "tilde prefix", spec: "openrouter/~deepseek/x", wantProvider: "openrouter", wantModel: "~deepseek/x"},
		{name: "nested slashes", spec: "groq/qwen/qwen3-32b", wantProvider: "groq", wantModel: "qwen/qwen3-32b"},
		{name: "bang is not a separator", spec: "openai/gpt-5!none", wantProvider: providerOpenAI, wantModel: "gpt-5!none"},
		{name: "colon tag", spec: "ollama/qwen3-embedding:0.6b", wantProvider: providerOllama, wantModel: "qwen3-embedding:0.6b"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			prov, model, err := parseModelString(tt.spec)

			require.NoError(t, err)
			assert.Equal(t, tt.wantProvider, prov.Name)
			assert.Equal(t, tt.wantModel, model)
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
