package smolllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseUsageReadsEveryProviderSpelling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		frame string
		want  Usage
	}{
		{
			name:  "plain counts",
			frame: `{"prompt_tokens":100,"completion_tokens":40}`,
			want:  newUsage(100, 40, 0, 0, false),
		},
		{
			name:  "deepseek prompt_cache_hit_tokens",
			frame: `{"prompt_tokens":100,"completion_tokens":40,"prompt_cache_hit_tokens":70}`,
			want:  newUsage(30, 40, 70, 0, false),
		},
		{
			name:  "openai prompt_tokens_details.cached_tokens",
			frame: `{"prompt_tokens":100,"completion_tokens":40,"prompt_tokens_details":{"cached_tokens":70}}`,
			want:  newUsage(30, 40, 70, 0, false),
		},
		{
			name: "reasoning tokens are a subset of output",
			frame: `{"prompt_tokens":10,"completion_tokens":40,` +
				`"completion_tokens_details":{"reasoning_tokens":25}}`,
			want: newUsage(10, 40, 0, 25, false),
		},
		{
			name: "deepseek wins when both cache spellings appear",
			frame: `{"prompt_tokens":100,"completion_tokens":5,"prompt_cache_hit_tokens":60,` +
				`"prompt_tokens_details":{"cached_tokens":10}}`,
			want: newUsage(40, 5, 60, 0, false),
		},
		{
			name:  "disagreeing counts never yield a negative input",
			frame: `{"prompt_tokens":10,"completion_tokens":1,"prompt_cache_hit_tokens":50}`,
			want:  newUsage(0, 1, 50, 0, false),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var chunk usageChunk
			require.NoError(t, json.Unmarshal([]byte(tt.frame), &chunk))

			got := parseUsage(chunk)
			assert.Equal(t, tt.want, got)
			// Total is derived, never taken from the provider.
			assert.Equal(t, got.Input+got.Output+got.CacheRead, got.Total)
			assert.False(t, got.Estimated)
		})
	}
}

// ADR-0002: usage stops at tokens. No cost field, no price source, ever.
func TestUsageCarriesNoCostField(t *testing.T) {
	t.Parallel()

	fields := reflect.VisibleFields(reflect.TypeOf(newUsage(0, 0, 0, 0, false)))
	names := make([]string, 0, len(fields))
	for _, field := range fields {
		names = append(names, field.Name)
		assert.NotContains(t, strings.ToLower(field.Name), "cost",
			"usage stops at tokens; pricing is the caller's concern")
		assert.NotContains(t, strings.ToLower(field.Name), "price")
	}
	assert.Equal(t, []string{"Input", "Output", "CacheRead", "Reasoning", "Total", "Estimated"}, names)

	encoded, err := json.Marshal(newUsage(1, 2, 3, 4, false))
	require.NoError(t, err)
	assert.NotContains(t, string(encoded), "cost")
}

func TestEstimateUsageIsMarked(t *testing.T) {
	t.Parallel()

	usage := estimateUsage(120, "some generated answer text")
	assert.Equal(t, 120, usage.Input)
	assert.Positive(t, usage.Output)
	assert.Equal(t, 0, usage.CacheRead)
	assert.Equal(t, usage.Input+usage.Output, usage.Total)
	assert.True(t, usage.Estimated, "a heuristic count must always say so")
}

func TestAskReportsCacheAndReasoningTokens(t *testing.T) {
	t.Parallel()

	usageFrame := `data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":40,` +
		`"prompt_cache_hit_tokens":70,"completion_tokens_details":{"reasoning_tokens":25}}}` + "\n\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte(usageFrame))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("deepseek/deepseek-v4-flash"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireAnswered(t, msg)

	assert.Equal(t, 30, msg.Usage.Input, "input excludes the cached tokens")
	assert.Equal(t, 40, msg.Usage.Output, "output includes the reasoning tokens")
	assert.Equal(t, 70, msg.Usage.CacheRead)
	assert.Equal(t, 25, msg.Usage.Reasoning)
	assert.Equal(t, 140, msg.Usage.Total)
	assert.False(t, msg.Usage.Estimated)
}

func TestAskMarksUsageEstimatedWhenProviderSendsNone(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeChatSuccess(t, w, "hello", "stop")
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	requireAnswered(t, msg)

	assert.True(t, msg.Usage.Estimated)
	assert.Equal(t, msg.Usage.Input+msg.Usage.Output+msg.Usage.CacheRead, msg.Usage.Total)
}

// A failed attempt serializes with its cause's message, not an empty object.
func TestAttemptErrorSerializesItsCause(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "quota exhausted", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	require.Equal(t, StopReasonError, msg.StopReason)
	require.Len(t, msg.Attempts, 1)

	encoded, err := json.Marshal(msg.Attempts[0])
	require.NoError(t, err)
	assert.Contains(t, string(encoded), "quota exhausted")
	assert.Contains(t, string(encoded), `"disposition":"advance"`)
	assert.Contains(t, string(encoded), `"status_code":429`)
}
