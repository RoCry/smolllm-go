package smolllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildEmbeddingURL(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		base     string
		provider string
		want     string
	}{
		{"plain base", "http://x", "openai", "http://x/v1/embeddings"},
		{"with version suffix", "http://x/v1", "openai", "http://x/v1/embeddings"},
		{"trailing slash", "http://x/", "openai", "http://x/embeddings"},
		{"trailing hash stripped", "http://x/custom/endpoint#", "openai", "http://x/custom/endpoint"},
		{"ollama default", "http://localhost:11434", "ollama", "http://localhost:11434/v1/embeddings"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got := resolveEndpointURL(tt.base, tt.provider, "embeddings")
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestEmbedPayloadShape(t *testing.T) {
	t.Parallel()

	t.Run("single input serializes as string", func(t *testing.T) {
		t.Parallel()
		req := embeddingRequest{Model: "test-model", Input: "hello", Dimensions: 0, ReasoningEffort: nil}
		data, err := json.Marshal(req)
		require.NoError(t, err)
		var m map[string]any
		require.NoError(t, json.Unmarshal(data, &m))
		assert.Equal(t, "hello", m["input"])
		assert.Equal(t, "test-model", m["model"])
		_, hasDim := m["dimensions"]
		assert.False(t, hasDim, "dimensions should be omitted when zero")
	})

	t.Run("multi input serializes as array", func(t *testing.T) {
		t.Parallel()
		req := embeddingRequest{Model: "test-model", Input: []string{"a", "b"}, Dimensions: 0, ReasoningEffort: nil}
		data, err := json.Marshal(req)
		require.NoError(t, err)
		var m map[string]any
		require.NoError(t, json.Unmarshal(data, &m))
		arr, ok := m["input"].([]any)
		require.True(t, ok, "input should be array")
		assert.Len(t, arr, 2)
		assert.Equal(t, "a", arr[0])
		assert.Equal(t, "b", arr[1])
	})

	t.Run("dimensions included when positive", func(t *testing.T) {
		t.Parallel()
		req := embeddingRequest{Model: "m", Input: "x", Dimensions: 256, ReasoningEffort: nil}
		data, err := json.Marshal(req)
		require.NoError(t, err)
		var m map[string]any
		require.NoError(t, json.Unmarshal(data, &m))
		assert.EqualValues(t, 256, m["dimensions"])
	})
}

func TestEmbedSendsDimensions(t *testing.T) {
	t.Parallel()

	var captured map[string]any
	var decodeErr error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeErr = json.NewDecoder(r.Body).Decode(&captured)
		resp := `{"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],` +
			` "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer srv.Close()

	_, err := Embed(context.Background(), []string{"hello"},
		WithModel("openai/test"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("k"),
		WithDimensions(128),
	)
	require.NoError(t, err)
	require.NoError(t, decodeErr)
	assert.EqualValues(t, 128, captured["dimensions"])
}

func TestEmbedUsesReasoningEffortSuffix(t *testing.T) {
	t.Parallel()

	var captured map[string]any
	var decodeErr error
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decodeErr = json.NewDecoder(r.Body).Decode(&captured)
		resp := `{"data": [{"index": 0, "embedding": [0.1, 0.2]}],` +
			` "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer srv.Close()

	resp, err := Embed(context.Background(), []string{"hello"},
		WithModel("openai/test-embedding!none"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("k"),
	)
	require.NoError(t, err)
	require.NoError(t, decodeErr)
	assert.Equal(t, "openai/test-embedding", resp.Model)
	assert.Equal(t, "test-embedding", resp.ModelName)
	assert.Equal(t, "test-embedding", captured["model"])
	assert.Equal(t, "none", captured["reasoning_effort"])
}

func TestEmbedRejectsEmptyInput(t *testing.T) {
	t.Parallel()

	t.Run("nil input", func(t *testing.T) {
		t.Parallel()
		_, err := Embed(context.Background(), nil, WithModel("openai/test"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "input must not be empty")
	})

	t.Run("empty slice", func(t *testing.T) {
		t.Parallel()
		_, err := Embed(context.Background(), []string{}, WithModel("openai/test"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "input must not be empty")
	})
}

func TestEmbedParsesResponseInInputOrder(t *testing.T) {
	t.Parallel()
	// Server returns data out of order by index.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{
			"object": "list",
			"data": [
				{"index": 1, "embedding": [0.2, 0.3]},
				{"index": 0, "embedding": [0.1, 0.4]}
			],
			"model": "test-model",
			"usage": {"prompt_tokens": 5, "total_tokens": 5}
		}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer srv.Close()

	resp, err := Embed(context.Background(), []string{"first", "second"},
		WithModel("openai/test-model"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("test-key"),
	)
	require.NoError(t, err)
	require.Len(t, resp.Embeddings, 2)
	// After sorting by index: index 0 → [0.1, 0.4], index 1 → [0.2, 0.3]
	assert.Equal(t, []float64{0.1, 0.4}, resp.Embeddings[0])
	assert.Equal(t, []float64{0.2, 0.3}, resp.Embeddings[1])
}

func TestEmbedMalformedResponseErrors(t *testing.T) {
	t.Parallel()

	t.Run("non-JSON body", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte("not json"))
		}))
		defer srv.Close()

		_, err := Embed(context.Background(), []string{"hello"},
			WithModel("openai/test"),
			WithBaseURL(srv.URL+"/"),
			WithAPIKey("k"),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "decode embedding response")
	})

	t.Run("mismatched data count", func(t *testing.T) {
		t.Parallel()
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			resp := `{"data": [{"index": 0, "embedding": [0.1]}],` +
				` "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1}}`
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(resp))
		}))
		defer srv.Close()

		_, err := Embed(context.Background(), []string{"a", "b"},
			WithModel("openai/test"),
			WithBaseURL(srv.URL+"/"),
			WithAPIKey("k"),
		)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "2 inputs")
	})
}

func TestEmbedHTTPErrorRetries(t *testing.T) {
	t.Parallel()
	var calls atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := calls.Add(1)
		if n == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte("rate limited"))
			return
		}
		resp := `{"data": [{"index": 0, "embedding": [0.5]}],` +
			` "model": "m", "usage": {"prompt_tokens": 2, "total_tokens": 2}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer srv.Close()

	resp, err := Embed(context.Background(), []string{"hi"},
		WithModel("openai/test"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("k"),
		WithTimeout(0), // no timeout — let retries run
	)
	require.NoError(t, err)
	require.Len(t, resp.Embeddings, 1)
	assert.Equal(t, []float64{0.5}, resp.Embeddings[0])
	assert.GreaterOrEqual(t, int(calls.Load()), 2, "should have retried at least once")
}

func TestEmbedHookFired(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		resp := `{"data": [{"index": 0, "embedding": [1.0]}],` +
			` "model": "m", "usage": {"prompt_tokens": 10, "total_tokens": 10}}`
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(resp))
	}))
	defer srv.Close()

	var captured Usage
	hookCalled := false
	_, err := Embed(context.Background(), []string{"test"},
		WithModel("openai/test"),
		WithBaseURL(srv.URL+"/"),
		WithAPIKey("k"),
		WithHook(func(e RequestEvent) {
			hookCalled = true
			captured = e.Usage
		}),
	)
	require.NoError(t, err)
	assert.True(t, hookCalled, "hook should have been called")
	assert.Equal(t, 10, captured.InputTokens)
	assert.Equal(t, 0, captured.OutputTokens)
	assert.Equal(t, "openai", captured.Provider)
	fmt.Println("Hook captured usage:", captured)
}
