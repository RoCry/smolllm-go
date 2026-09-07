package smolllm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSequentialSelector(t *testing.T) {
	t.Parallel()

	t.Run("single model", func(t *testing.T) {
		t.Parallel()
		s := NewSequentialSelector([]string{"model1"})
		m, ok := s.NextModel()
		assert.True(t, ok)
		assert.Equal(t, "model1", m)
		_, ok = s.NextModel()
		assert.False(t, ok)
	})

	t.Run("multiple models", func(t *testing.T) {
		t.Parallel()
		s := NewSequentialSelector([]string{"m1", "m2", "m3"})
		m1, ok1 := s.NextModel()
		assert.True(t, ok1)
		assert.Equal(t, "m1", m1)
		m2, ok2 := s.NextModel()
		assert.True(t, ok2)
		assert.Equal(t, "m2", m2)
		m3, ok3 := s.NextModel()
		assert.True(t, ok3)
		assert.Equal(t, "m3", m3)
		_, ok4 := s.NextModel()
		assert.False(t, ok4)
	})

	t.Run("empty list", func(t *testing.T) {
		t.Parallel()
		s := NewSequentialSelector([]string{})
		_, ok := s.NextModel()
		assert.False(t, ok)
	})
}

func TestRandomSelector(t *testing.T) {
	t.Parallel()

	t.Run("exhausts all models", func(t *testing.T) {
		t.Parallel()
		models := []string{"a", "b", "c"}
		s := NewRandomSelector(models, nil)
		seen := make(map[string]bool)
		for i := 0; i < 3; i++ {
			m, ok := s.NextModel()
			require.True(t, ok)
			seen[m] = true
		}
		_, ok := s.NextModel()
		assert.False(t, ok)
		assert.Len(t, seen, 3)
		assert.True(t, seen["a"])
		assert.True(t, seen["b"])
		assert.True(t, seen["c"])
	})

	t.Run("with weights exhausts all", func(t *testing.T) {
		t.Parallel()
		models := []string{"high", "low"}
		weights := map[string]float64{"high": 9, "low": 1}
		s := NewRandomSelector(models, weights)
		seen := make(map[string]bool)
		for i := 0; i < 2; i++ {
			m, ok := s.NextModel()
			require.True(t, ok)
			seen[m] = true
		}
		_, ok := s.NextModel()
		assert.False(t, ok)
		assert.Len(t, seen, 2)
	})

	t.Run("weighted distribution", func(t *testing.T) {
		t.Parallel()
		// high should be picked first ~90% of the time
		counts := map[string]int{"high": 0, "low": 0}
		trials := 1000
		for i := 0; i < trials; i++ {
			s := NewRandomSelector([]string{"high", "low"}, map[string]float64{"high": 9, "low": 1})
			first, _ := s.NextModel()
			counts[first]++
		}
		// Allow variance: high should be >80%
		assert.Greater(t, counts["high"], trials*8/10)
		assert.Less(t, counts["low"], trials*3/10)
	})
}

func TestCreateSelector(t *testing.T) {
	t.Run("from Model string", func(t *testing.T) {
		t.Parallel()
		opts := applyOptions(WithModel("a,b,c"))
		s, err := createSelector(opts)
		require.NoError(t, err)
		_, ok := s.(*SequentialSelector)
		assert.True(t, ok, "should be SequentialSelector")
	})

	t.Run("from explicit factory", func(t *testing.T) {
		t.Parallel()
		opts := applyOptions()
		opts.NewSelector = func() ModelSelector { return NewRandomSelector([]string{"x"}, nil) }
		s, err := createSelector(opts)
		require.NoError(t, err)
		m, ok := s.NextModel()
		require.True(t, ok)
		assert.Equal(t, "x", m)
	})

	t.Run("factory takes precedence over Model", func(t *testing.T) {
		t.Parallel()
		opts := applyOptions(WithModel("ignored"))
		opts.NewSelector = func() ModelSelector { return NewRandomSelector([]string{"x"}, nil) }
		s, err := createSelector(opts)
		require.NoError(t, err)
		m, _ := s.NextModel()
		assert.Equal(t, "x", m)
	})

	// The bug this guards: a selector hands each model out once, so a shared
	// instance would leave the second call with an empty pool.
	t.Run("every call gets a fresh selector", func(t *testing.T) {
		t.Parallel()
		for _, opts := range []Options{
			applyOptions(WithModelSet("a", "b")),
			applyOptions(WithModelWeights(map[string]float64{"a": 1, "b": 2})),
			applyOptions(WithModel("a,b")),
		} {
			first, err := createSelector(opts)
			require.NoError(t, err)
			drained := 0
			for {
				if _, ok := first.NextModel(); !ok {
					break
				}
				drained++
			}
			require.Equal(t, 2, drained)

			second, err := createSelector(opts)
			require.NoError(t, err)
			_, ok := second.NextModel()
			assert.True(t, ok, "draining one selector must not exhaust the next")
		}
	})

	t.Run("empty model string error", func(t *testing.T) {
		t.Setenv("SMOLLLM_MODEL", "")
		opts := applyOptions(WithModel(""))
		_, err := createSelector(opts)
		require.Error(t, err)
	})

	t.Run("model with empty entry error", func(t *testing.T) {
		t.Parallel()
		opts := applyOptions(WithModel("a,,b"))
		_, err := createSelector(opts)
		require.Error(t, err)
	})
}

func TestWithModelSetOption(t *testing.T) {
	t.Parallel()
	opts := applyOptions(WithModelSet("a", "b", "c"))
	require.NotNil(t, opts.NewSelector)
	_, ok := opts.NewSelector().(*RandomSelector)
	assert.True(t, ok)
}

func TestWithModelSetDoesNotAliasCallerSlice(t *testing.T) {
	t.Parallel()
	models := []string{"a", "b"}
	opts := applyOptions(WithModelSet(models...))
	models[0] = testMutated

	selector := opts.NewSelector()
	seen := map[string]bool{}
	for {
		m, ok := selector.NextModel()
		if !ok {
			break
		}
		seen[m] = true
	}
	assert.True(t, seen["a"])
	assert.False(t, seen[testMutated])
}

func TestWithModelWeightsOption(t *testing.T) {
	t.Parallel()
	opts := applyOptions(WithModelWeights(map[string]float64{"a": 1, "b": 2}))
	require.NotNil(t, opts.NewSelector)
	_, ok := opts.NewSelector().(*RandomSelector)
	assert.True(t, ok)
}

func TestWithModelSetPanic(t *testing.T) {
	t.Parallel()
	assert.Panics(t, func() {
		WithModelSet()
	})
}

func TestWithModelWeightsPanic(t *testing.T) {
	t.Parallel()
	t.Run("empty map", func(t *testing.T) {
		t.Parallel()
		assert.Panics(t, func() {
			WithModelWeights(map[string]float64{})
		})
	})
	t.Run("zero weight", func(t *testing.T) {
		t.Parallel()
		assert.Panics(t, func() {
			WithModelWeights(map[string]float64{"a": 0})
		})
	})
	t.Run("negative weight", func(t *testing.T) {
		t.Parallel()
		assert.Panics(t, func() {
			WithModelWeights(map[string]float64{"a": -1})
		})
	})
}

// Validate drains a selector. Before the fix it drained the *shared* one stored
// in Options, so every later call on the same Client found an empty pool and
// failed instantly with "no models were attempted" without touching the network.
func TestValidateDoesNotExhaustLaterCalls(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		option Option
	}{
		{name: "WithModelSet", option: WithModelSet("openai/model-a", "openai/model-b")},
		{
			name:   "WithModelWeights",
			option: WithModelWeights(map[string]float64{"openai/model-a": 1, "openai/model-b": 2}),
		},
		// The comma chain is what agentiu uses for ordered fallback.
		{name: "WithModel comma chain", option: WithModel("openai/model-a,openai/model-b")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var requests atomic.Int32
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				requests.Add(1)
				writeChatSuccess(t, w, "hello", "stop")
			}))
			defer srv.Close()

			client := New(tt.option, withTestProvider(srv.URL+"/", "test-key"))

			require.NoError(t, client.Validate())
			assert.Equal(t, int32(0), requests.Load(), "Validate is offline")

			// Two calls after Validate: both must reach a provider.
			for i := range 2 {
				msg := client.Ask(context.Background(), RequestFromString("hi"))
				requireAnswered(t, msg)
				assert.Equal(t, "hello", msg.Content)
				assert.Equal(t, int32(i+1), requests.Load(), "call %d must attempt a leg", i+1)
			}
		})
	}
}

// The same hazard reaches Stream and Embed, which share createSelector.
func TestValidateDoesNotExhaustStreamOrEmbed(t *testing.T) {
	t.Parallel()

	var chatRequests atomic.Int32
	var embedRequests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/embeddings") {
			embedRequests.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1]}],"model":"m","usage":{"prompt_tokens":1}}`))
			return
		}
		chatRequests.Add(1)
		writeChatSuccess(t, w, "hello", "stop")
	}))
	defer srv.Close()

	client := New(
		WithModelSet("openai/model-a", "openai/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	)
	require.NoError(t, client.Validate())

	msg := drain(client.Stream(context.Background(), RequestFromString("hi")))
	requireAnswered(t, msg)
	assert.Equal(t, int32(1), chatRequests.Load())

	resp, err := client.Embed(context.Background(), []string{"hi"})
	require.NoError(t, err)
	require.Len(t, resp.Embeddings, 1)
	assert.Equal(t, int32(1), embedRequests.Load())
}

// Two calls on one Client must not share selector state even without Validate.
func TestConcurrentCallsDoNotShareSelectorState(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		writeChatSuccess(t, w, "hello", "stop")
	}))
	defer srv.Close()

	client := New(
		WithModelSet("openai/model-a", "openai/model-b"),
		withTestProvider(srv.URL+"/", "test-key"),
	)

	var wg sync.WaitGroup
	results := make([]*AssistantMessage, 8)
	for i := range results {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i] = client.Ask(context.Background(), RequestFromString("hi"))
		}()
	}
	wg.Wait()

	for _, msg := range results {
		requireAnswered(t, msg)
		assert.Equal(t, "hello", msg.Content)
	}
	assert.Equal(t, int32(len(results)), requests.Load())
}
