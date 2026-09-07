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
		expected delta
	}{
		{"delta", `data: {"choices":[{"delta":{"content":"hello"}}]}`, delta{Content: "hello", Reasoning: ""}},
		{"done", "data: [DONE]", delta{Content: "", Reasoning: ""}},
		{
			"reasoning_content (DeepSeek)",
			`data: {"choices":[{"delta":{"content":"","reasoning_content":"thinking..."}}]}`,
			delta{Content: "", Reasoning: "thinking..."},
		},
		{
			"reasoning (Ollama)",
			`data: {"choices":[{"delta":{"content":"answer","reasoning":"thought"}}]}`,
			delta{Content: "answer", Reasoning: "thought"},
		},
		{
			"reasoning_content takes precedence",
			`data: {"choices":[{"delta":{"content":"","reasoning_content":"rc","reasoning":"r"}}]}`,
			delta{Content: "", Reasoning: "rc"},
		},
	}

	logger := slog.New(slog.DiscardHandler)

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			actual, err := parseChunkLine(logger, tc.line, nil, nil, nil, nil)
			require.NoError(t, err)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestProcessChunkLineUpdatesUsage(t *testing.T) {
	t.Parallel()

	logger := slog.New(slog.DiscardHandler)
	usage := &reportedUsage{usage: Usage{}, reported: false} //nolint:exhaustruct // overwritten by the parser
	actual, err := parseChunkLine(
		logger,
		`data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`,
		usage,
		nil,
		nil,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, delta{Content: "", Reasoning: ""}, actual)
	require.True(t, usage.reported)
	assert.Equal(t, 11, usage.usage.Input)
	assert.Equal(t, 7, usage.usage.Output)
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
		assert.Empty(t, reasoning)
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
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		result := f.Feed(delta{Content: "<think>thought</think>answer", Reasoning: ""})
		assert.Equal(t, "thought", result.Reasoning)
		assert.Equal(t, "answer", result.Content)
	})

	t.Run("split across chunks", func(t *testing.T) {
		t.Parallel()
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		r1 := f.Feed(delta{Content: "<think>tho", Reasoning: ""})
		assert.Equal(t, "tho", r1.Reasoning)
		assert.Empty(t, r1.Content)

		r2 := f.Feed(delta{Content: "ught</think>answer", Reasoning: ""})
		assert.Equal(t, "ught", r2.Reasoning)
		assert.Equal(t, "answer", r2.Content)
	})

	t.Run("tag split at boundary", func(t *testing.T) {
		t.Parallel()
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		r1 := f.Feed(delta{Content: "<thi", Reasoning: ""})
		assert.Empty(t, r1.Content)
		assert.Empty(t, r1.Reasoning)

		r2 := f.Feed(delta{Content: "nk>reasoning</think>content", Reasoning: ""})
		assert.Equal(t, "reasoning", r2.Reasoning)
		assert.Equal(t, "content", r2.Content)
	})

	t.Run("closing tag split", func(t *testing.T) {
		t.Parallel()
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		r1 := f.Feed(delta{Content: "<think>thought</th", Reasoning: ""})
		assert.Equal(t, "thought", r1.Reasoning)

		r2 := f.Feed(delta{Content: "ink>answer", Reasoning: ""})
		assert.Empty(t, r2.Reasoning)
		assert.Equal(t, "answer", r2.Content)
	})

	t.Run("flush", func(t *testing.T) {
		t.Parallel()
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		r1 := f.Feed(delta{Content: "<think>partial", Reasoning: ""})
		assert.Equal(t, "partial", r1.Reasoning)

		r2 := f.Feed(delta{Content: "more</thi", Reasoning: ""})
		assert.Equal(t, "more", r2.Reasoning)

		result := f.Flush()
		assert.Equal(t, "</thi", result.Reasoning)
	})

	t.Run("flush empty", func(t *testing.T) {
		t.Parallel()
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		result := f.Flush()
		assert.True(t, result.IsEmpty())
	})

	t.Run("passthrough when backend provides reasoning", func(t *testing.T) {
		t.Parallel()
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		chunk := delta{Content: "<think>inline</think>text", Reasoning: "backend reasoning"}
		result := f.Feed(chunk)
		assert.Equal(t, "<think>inline</think>text", result.Content)
		assert.Equal(t, "backend reasoning", result.Reasoning)

		// Subsequent chunks should also pass through.
		chunk2 := delta{Content: "<think>more</think>stuff", Reasoning: ""}
		result2 := f.Feed(chunk2)
		assert.Equal(t, "<think>more</think>stuff", result2.Content)
		assert.Empty(t, result2.Reasoning)
	})

	t.Run("no think tags", func(t *testing.T) {
		t.Parallel()
		f := &thinkTagFilter{insideThink: false, buffer: "", disabled: false}
		result := f.Feed(delta{Content: "just content", Reasoning: ""})
		assert.Equal(t, "just content", result.Content)
		assert.Empty(t, result.Reasoning)
	})
}

func TestDeltaIsEmpty(t *testing.T) {
	t.Parallel()
	assert.True(t, (delta{Content: "", Reasoning: ""}).IsEmpty())
	assert.False(t, (delta{Content: "x", Reasoning: ""}).IsEmpty())
	assert.False(t, (delta{Content: "", Reasoning: "x"}).IsEmpty())
}

func TestBuildRequestURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name     string
		base     string
		provider string
		want     string
	}{
		{
			"openai default", "https://api.openai.com", "openai",
			"https://api.openai.com/v1/chat/completions",
		},
		{
			"gemini default", "https://generativelanguage.googleapis.com", "gemini",
			"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
		},
		{
			"anthropic trailing slash", "https://api.anthropic.com/", "anthropic",
			"https://api.anthropic.com/v1/chat/completions",
		},
		{
			"ollama trailing slash", "http://localhost:11434/", "ollama",
			"http://localhost:11434/chat/completions",
		},
		{
			"hash override", "http://localhost:1234#", "custom",
			"http://localhost:1234",
		},
		{
			"version suffix default", "https://ark.cn-beijing.volces.com/api/v3", "volcengine",
			"https://ark.cn-beijing.volces.com/api/v3/chat/completions",
		},
		{
			"version suffix v2", "https://example.com/v2", "openai",
			"https://example.com/v2/chat/completions",
		},
		{
			"anthropic with version", "https://api.anthropic.com/v1", "anthropic",
			"https://api.anthropic.com/v1/chat/completions",
		},
		{
			"anthropic without version", "https://api.anthropic.com", "anthropic",
			"https://api.anthropic.com/v1/chat/completions",
		},
		{
			"gemini with version", "https://generativelanguage.googleapis.com/v2", "gemini",
			"https://generativelanguage.googleapis.com/v2/chat/completions",
		},
		{
			"gemini without version", "https://generativelanguage.googleapis.com", "gemini",
			"https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
		},
		{
			"version suffix trailing slash", "https://ark.cn-beijing.volces.com/api/v3/", "volcengine",
			"https://ark.cn-beijing.volces.com/api/v3/chat/completions",
		},
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
