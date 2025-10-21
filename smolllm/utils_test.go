package smolllm

import (
	"math/rand"
	"testing"

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
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			require.Equal(t, tc.expected, stripBackticks(tc.input))
		})
	}
}

func TestResolveModels(t *testing.T) {
	t.Setenv("SMOLLLM_MODEL", "openai/gpt-4 , grok/grok-1")
	models, err := resolveModels("")
	require.NoError(t, err)
	require.Equal(t, []string{"openai/gpt-4", "grok/grok-1"}, models)
}

func TestResolveModelsErrors(t *testing.T) {
	t.Setenv("SMOLLLM_MODEL", "")
	_, err := resolveModels("")
	require.Error(t, err)
}

func TestBalancerChoosePair(t *testing.T) {
	t.Parallel()
	b := &simpleBalancer{
		usage: make(map[pairKey]int),
		rnd:   rand.New(rand.NewSource(1)),
	}

	cases := []struct {
		name    string
		keys    string
		urls    string
		wantErr bool
	}{
		{"single", "k1", "u1", false},
		{"multi key single url", "k1,k2", "u1", false},
		{"mismatch counts", "a,b", "u1,u2,u3", true},
		{"empty entry", "a,", "u1", true},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, _, err := b.choosePair(tc.keys, tc.urls)
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
		})
	}
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

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			actual, err := processChunkLine(tc.line)
			require.NoError(t, err)
			require.Equal(t, tc.expected, actual)
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
		{"https://api.anthropic.com/", "anthropic", "https://api.anthropic.com/v1"},
		{"http://localhost:11434/", "ollama", "http://localhost:11434/chat/completions"},
		{"http://localhost:1234#", "custom", "http://localhost:1234"},
	}

	for _, tc := range cases {
		got := buildRequestURL(tc.base, tc.provider)
		require.Equal(t, tc.want, got)
	}
}
