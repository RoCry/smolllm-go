package smolllm

import (
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStripBackticks(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"trim markdown", "```markdown\nhello\n```", "hello"},
		{"unchanged", "plain text", "plain text"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, stripBackticks(tc.input))
		})
	}
}

func TestCreateSelectorFromEnv(t *testing.T) {
	t.Setenv("SMOLLLM_MODEL", "openai/gpt-4 , grok/grok-1")
	opts := applyOptions()
	selector, err := createSelector(opts)
	require.NoError(t, err)
	m1, ok1 := selector.NextModel()
	assert.True(t, ok1)
	assert.Equal(t, "openai/gpt-4", m1)
	m2, ok2 := selector.NextModel()
	assert.True(t, ok2)
	assert.Equal(t, "grok/grok-1", m2)
	_, ok3 := selector.NextModel()
	assert.False(t, ok3)
}

func TestCreateSelectorErrors(t *testing.T) {
	t.Setenv("SMOLLLM_MODEL", "")
	opts := applyOptions()
	_, err := createSelector(opts)
	require.Error(t, err)
}

func TestProcessChunkLine(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		line     string
		expected string
	}{
		{"delta", `data: {"choices":[{"delta":{"content":"hello"}}]}`, "hello"},
		{"done", "data: [DONE]", ""},
	}

	logger := slog.New(slog.DiscardHandler)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			actual, err := processChunkLine(logger, tc.line)
			require.NoError(t, err)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestBuildRequestURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		base     string
		provider string
		want     string
	}{
		{"https://api.openai.com", "openai", "https://api.openai.com/v1/chat/completions"},
		{"https://generativelanguage.googleapis.com", "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
		{"https://api.anthropic.com/", "anthropic", "https://api.anthropic.com/v1/chat/completions"},
		{"http://localhost:11434/", "ollama", "http://localhost:11434/chat/completions"},
		{"http://localhost:1234#", "custom", "http://localhost:1234"},
	}

	for _, tc := range cases {
		got := buildRequestURL(tc.base, tc.provider)
		assert.Equal(t, tc.want, got)
	}
}
