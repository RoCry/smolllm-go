package smolllm

import (
	"math/rand"
	"testing"
)

func TestStripBackticks(t *testing.T) {
	t.Parallel()
	input := "```markdown\nhello\n```"
	want := "hello"
	got := stripBackticks(input)
	if got != want {
		t.Fatalf("stripBackticks(%q)=%q, want %q", input, got, want)
	}

	unchanged := "plain text"
	if out := stripBackticks(unchanged); out != unchanged {
		t.Fatalf("expected unchanged string, got %q", out)
	}
}

func TestResolveModels(t *testing.T) {
	t.Setenv("SMOLLLM_MODEL", "openai/gpt-4 , grok/grok-1")
	models, err := resolveModels("")
	if err != nil {
		t.Fatalf("resolveModels returned error: %v", err)
	}
	want := []string{"openai/gpt-4", "grok/grok-1"}
	if len(models) != len(want) {
		t.Fatalf("unexpected model count %d, want %d", len(models), len(want))
	}
	for i, model := range models {
		if model != want[i] {
			t.Fatalf("model[%d]=%q want %q", i, model, want[i])
		}
	}
}

func TestResolveModelsErrors(t *testing.T) {
	t.Setenv("SMOLLLM_MODEL", "")
	if _, err := resolveModels(""); err == nil {
		t.Fatalf("expected error when no model configured")
	}
}

func TestBalancerChoosePair(t *testing.T) {
	t.Parallel()
	b := &simpleBalancer{
		usage: make(map[pairKey]int),
		rnd:   rand.New(rand.NewSource(1)),
	}

	key, url, err := b.choosePair("k1", "u1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if key != "k1" || url != "u1" {
		t.Fatalf("expected (k1,u1), got (%s,%s)", key, url)
	}

	_, _, err = b.choosePair("k1,k2", "u1")
	if err != nil {
		t.Fatalf("unexpected error for mismatched counts with single url: %v", err)
	}

	if _, _, err := b.choosePair("a,b", "u1,u2,u3"); err == nil {
		t.Fatalf("expected mismatch error")
	}

	if _, _, err := b.choosePair("a,", "u1"); err == nil {
		t.Fatalf("expected empty entry error")
	}
}

func TestProcessChunkLine(t *testing.T) {
	t.Parallel()
	line := `data: {"choices":[{"delta":{"content":"hello"}}]}`
	got, err := processChunkLine(line)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "hello" {
		t.Fatalf("unexpected content %q", got)
	}

	done, err := processChunkLine("data: [DONE]")
	if err != nil {
		t.Fatalf("unexpected error for done line: %v", err)
	}
	if done != "" {
		t.Fatalf("expected empty string for done line, got %q", done)
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
		if got != tc.want {
			t.Fatalf("buildRequestURL(%q,%q)=%q want %q", tc.base, tc.provider, got, tc.want)
		}
	}
}
