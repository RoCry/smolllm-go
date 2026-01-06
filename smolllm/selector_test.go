package smolllm

import (
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

	t.Run("from explicit Selector", func(t *testing.T) {
		t.Parallel()
		explicit := NewRandomSelector([]string{"x"}, nil)
		opts := applyOptions()
		opts.Selector = explicit
		s, err := createSelector(opts)
		require.NoError(t, err)
		assert.Equal(t, explicit, s)
	})

	t.Run("Selector takes precedence over Model", func(t *testing.T) {
		t.Parallel()
		explicit := NewRandomSelector([]string{"x"}, nil)
		opts := applyOptions(WithModel("ignored"))
		opts.Selector = explicit
		s, err := createSelector(opts)
		require.NoError(t, err)
		m, _ := s.NextModel()
		assert.Equal(t, "x", m)
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
	require.NotNil(t, opts.Selector)
	_, ok := opts.Selector.(*RandomSelector)
	assert.True(t, ok)
}

func TestWithModelWeightsOption(t *testing.T) {
	t.Parallel()
	opts := applyOptions(WithModelWeights(map[string]float64{"a": 1, "b": 2}))
	require.NotNil(t, opts.Selector)
	_, ok := opts.Selector.(*RandomSelector)
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
