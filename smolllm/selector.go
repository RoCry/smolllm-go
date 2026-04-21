package smolllm

import (
	"errors"
	"math/rand/v2"
	"os"
	"strings"
)

// ModelSelector defines how models are chosen during fallback.
type ModelSelector interface {
	// NextModel returns the next model to try. Returns empty string and false when exhausted.
	NextModel() (string, bool)
	// HasMore returns true if there are more models to try after the current one.
	HasMore() bool
}

// SequentialSelector tries models in order. Used for comma-separated strings and slices.
type SequentialSelector struct {
	models []string
	idx    int
}

// NewSequentialSelector creates a selector that tries models in order.
func NewSequentialSelector(models []string) *SequentialSelector {
	return &SequentialSelector{models: models, idx: 0}
}

// NextModel returns the next model in sequence, or empty string and false when exhausted.
func (s *SequentialSelector) NextModel() (string, bool) {
	if s.idx >= len(s.models) {
		return "", false
	}
	model := s.models[s.idx]
	s.idx++
	return model, true
}

// HasMore reports whether additional models remain in the sequence.
func (s *SequentialSelector) HasMore() bool {
	return s.idx < len(s.models)
}

// RandomSelector picks models randomly with optional weights.
// On each call, it removes the chosen model from the pool.
type RandomSelector struct {
	weights   map[string]float64
	remaining []string
}

// NewRandomSelector creates a selector with weighted random selection.
// Pass nil or empty map weights to use equal weights for all models.
func NewRandomSelector(models []string, weights map[string]float64) *RandomSelector {
	w := make(map[string]float64, len(models))
	for _, m := range models {
		if weights != nil {
			if wt, ok := weights[m]; ok {
				w[m] = wt
				continue
			}
		}
		w[m] = 1.0
	}
	remaining := make([]string, len(models))
	copy(remaining, models)
	return &RandomSelector{weights: w, remaining: remaining}
}

// NextModel picks a weighted-random model from the remaining pool, removes it, and returns it.
func (r *RandomSelector) NextModel() (string, bool) {
	if len(r.remaining) == 0 {
		return "", false
	}

	// Build cumulative weights
	total := 0.0
	for _, m := range r.remaining {
		total += r.weights[m]
	}

	// Pick random point
	point := rand.Float64() * total
	cumulative := 0.0
	chosenIdx := len(r.remaining) - 1 // Default to last to avoid bias from float rounding
	for i, m := range r.remaining {
		cumulative += r.weights[m]
		if point <= cumulative {
			chosenIdx = i
			break
		}
	}

	chosen := r.remaining[chosenIdx]
	// Remove chosen from remaining
	r.remaining = append(r.remaining[:chosenIdx], r.remaining[chosenIdx+1:]...)
	return chosen, true
}

// HasMore reports whether the random pool still contains models.
func (r *RandomSelector) HasMore() bool {
	return len(r.remaining) > 0
}

// createSelector creates the appropriate selector based on Options.
// Priority: Options.Selector > Options.Model > SMOLLLM_MODEL env.
func createSelector(opts Options) (ModelSelector, error) {
	// If explicit selector set, use it
	if opts.Selector != nil {
		return opts.Selector, nil
	}

	// Fall back to Model string (comma-separated sequential)
	candidate := strings.TrimSpace(opts.Model)
	if candidate == "" {
		candidate = os.Getenv("SMOLLLM_MODEL")
	}
	if strings.TrimSpace(candidate) == "" {
		return nil, errors.New("model string not provided. set SMOLLLM_MODEL or call WithModel")
	}

	parts := strings.Split(candidate, ",")
	models := make([]string, 0, len(parts))
	for _, part := range parts {
		value := strings.TrimSpace(part)
		if value == "" {
			return nil, errors.New("model string contains empty entry")
		}
		models = append(models, value)
	}

	return NewSequentialSelector(models), nil
}
