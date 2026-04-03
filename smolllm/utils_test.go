package smolllm

import (
	"fmt"
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
		expected StreamChunk
	}{
		{"delta", `data: {"choices":[{"delta":{"content":"hello"}}]}`, StreamChunk{Content: "hello"}},
		{"done", "data: [DONE]", StreamChunk{}},
		{
			"reasoning_content (DeepSeek)",
			`data: {"choices":[{"delta":{"content":"","reasoning_content":"thinking..."}}]}`,
			StreamChunk{Reasoning: "thinking..."},
		},
		{
			"reasoning (Ollama)",
			`data: {"choices":[{"delta":{"content":"answer","reasoning":"thought"}}]}`,
			StreamChunk{Content: "answer", Reasoning: "thought"},
		},
		{
			"reasoning_content takes precedence",
			`data: {"choices":[{"delta":{"content":"","reasoning_content":"rc","reasoning":"r"}}]}`,
			StreamChunk{Reasoning: "rc"},
		},
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

func TestExtractThinkTags(t *testing.T) {
	t.Parallel()

	t.Run("basic", func(t *testing.T) {
		t.Parallel()
		reasoning, content := extractThinkTags("<think>reasoning</think>answer")
		assert.Equal(t, "reasoning", reasoning)
		assert.Equal(t, "answer", content)
	})

	t.Run("multiple blocks", func(t *testing.T) {
		t.Parallel()
		reasoning, content := extractThinkTags("<think>first</think>middle<think>second</think>end")
		assert.Equal(t, "first\n\nsecond", reasoning)
		assert.Equal(t, "middleend", content)
	})

	t.Run("no think tags", func(t *testing.T) {
		t.Parallel()
		reasoning, content := extractThinkTags("just plain text")
		assert.Equal(t, "", reasoning)
		assert.Equal(t, "just plain text", content)
	})

	t.Run("multiline think", func(t *testing.T) {
		t.Parallel()
		reasoning, content := extractThinkTags("<think>\nline1\nline2\n</think>answer")
		assert.Contains(t, reasoning, "line1")
		assert.Contains(t, reasoning, "line2")
		assert.Equal(t, "answer", content)
	})
}

func TestThinkTagFilter(t *testing.T) {
	t.Parallel()

	t.Run("basic", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		result := f.Feed(StreamChunk{Content: "<think>thought</think>answer"})
		assert.Equal(t, "thought", result.Reasoning)
		assert.Equal(t, "answer", result.Content)
	})

	t.Run("split across chunks", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		r1 := f.Feed(StreamChunk{Content: "<think>tho"})
		assert.Equal(t, "tho", r1.Reasoning)
		assert.Equal(t, "", r1.Content)

		r2 := f.Feed(StreamChunk{Content: "ught</think>answer"})
		assert.Equal(t, "ught", r2.Reasoning)
		assert.Equal(t, "answer", r2.Content)
	})

	t.Run("tag split at boundary", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		r1 := f.Feed(StreamChunk{Content: "<thi"})
		assert.Equal(t, "", r1.Content)
		assert.Equal(t, "", r1.Reasoning)

		r2 := f.Feed(StreamChunk{Content: "nk>reasoning</think>content"})
		assert.Equal(t, "reasoning", r2.Reasoning)
		assert.Equal(t, "content", r2.Content)
	})

	t.Run("closing tag split", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		r1 := f.Feed(StreamChunk{Content: "<think>thought</th"})
		assert.Equal(t, "thought", r1.Reasoning)

		r2 := f.Feed(StreamChunk{Content: "ink>answer"})
		assert.Equal(t, "", r2.Reasoning)
		assert.Equal(t, "answer", r2.Content)
	})

	t.Run("flush", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		r1 := f.Feed(StreamChunk{Content: "<think>partial"})
		assert.Equal(t, "partial", r1.Reasoning)

		r2 := f.Feed(StreamChunk{Content: "more</thi"})
		assert.Equal(t, "more", r2.Reasoning)

		result := f.Flush()
		assert.Equal(t, "</thi", result.Reasoning)
	})

	t.Run("flush empty", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		result := f.Flush()
		assert.True(t, result.IsEmpty())
	})

	t.Run("passthrough when backend provides reasoning", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		chunk := StreamChunk{Content: "<think>inline</think>text", Reasoning: "backend reasoning"}
		result := f.Feed(chunk)
		assert.Equal(t, "<think>inline</think>text", result.Content)
		assert.Equal(t, "backend reasoning", result.Reasoning)

		// Subsequent chunks should also pass through.
		chunk2 := StreamChunk{Content: "<think>more</think>stuff"}
		result2 := f.Feed(chunk2)
		assert.Equal(t, "<think>more</think>stuff", result2.Content)
		assert.Equal(t, "", result2.Reasoning)
	})

	t.Run("no think tags", func(t *testing.T) {
		t.Parallel()
		f := &ThinkTagFilter{}
		result := f.Feed(StreamChunk{Content: "just content"})
		assert.Equal(t, "just content", result.Content)
		assert.Equal(t, "", result.Reasoning)
	})
}

func TestStreamChunkString(t *testing.T) {
	t.Parallel()
	chunk := StreamChunk{Content: "hello", Reasoning: "thinking"}
	assert.Equal(t, "hello", chunk.String())
	assert.Equal(t, "hello", fmt.Sprint(chunk))
}

func TestStreamChunkIsEmpty(t *testing.T) {
	t.Parallel()
	assert.True(t, StreamChunk{}.IsEmpty())
	assert.False(t, StreamChunk{Content: "x"}.IsEmpty())
	assert.False(t, StreamChunk{Reasoning: "x"}.IsEmpty())
}

func TestBuildRequestURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		base     string
		provider string
		want     string
	}{
		{"openai default", "https://api.openai.com", "openai", "https://api.openai.com/v1/chat/completions"},
		{"gemini default", "https://generativelanguage.googleapis.com", "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
		{"anthropic trailing slash", "https://api.anthropic.com/", "anthropic", "https://api.anthropic.com/v1/chat/completions"},
		{"ollama trailing slash", "http://localhost:11434/", "ollama", "http://localhost:11434/chat/completions"},
		{"hash override", "http://localhost:1234#", "custom", "http://localhost:1234"},
		// Version suffix: base URL already contains /v3
		{"version suffix default", "https://ark.cn-beijing.volces.com/api/v3", "volcengine", "https://ark.cn-beijing.volces.com/api/v3/chat/completions"},
		// Higher version number
		{"version suffix v2", "https://example.com/v2", "openai", "https://example.com/v2/chat/completions"},
		// Anthropic with version suffix
		{"anthropic with version", "https://api.anthropic.com/v1", "anthropic", "https://api.anthropic.com/v1/chat/completions"},
		// Anthropic without version suffix (default behavior)
		{"anthropic without version", "https://api.anthropic.com", "anthropic", "https://api.anthropic.com/v1/chat/completions"},
		// Gemini with version suffix
		{"gemini with version", "https://generativelanguage.googleapis.com/v2", "gemini", "https://generativelanguage.googleapis.com/v2/chat/completions"},
		// Gemini without version suffix (default behavior)
		{"gemini without version", "https://generativelanguage.googleapis.com", "gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
		// Version suffix with trailing slash
		{"version suffix trailing slash", "https://ark.cn-beijing.volces.com/api/v3/", "volcengine", "https://ark.cn-beijing.volces.com/api/v3/chat/completions"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := buildRequestURL(tc.base, tc.provider)
			assert.Equal(t, tc.want, got)
		})
	}
}

func TestHasVersionSuffix(t *testing.T) {
	t.Parallel()
	cases := []struct {
		url  string
		want bool
	}{
		{"https://api.openai.com", false},
		{"https://ark.cn-beijing.volces.com/api/v3", true},
		{"https://example.com/v1", true},
		{"https://example.com/v2", true},
		{"https://example.com/v123", true},
		{"https://example.com/v1/", true},
		{"https://example.com/v1beta", false},
		{"https://example.com/version1", false},
	}

	for _, tc := range cases {
		t.Run(tc.url, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.want, hasVersionSuffix(tc.url))
		})
	}
}
