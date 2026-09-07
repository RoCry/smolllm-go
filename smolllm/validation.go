package smolllm

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// Validate ensures that all model/provider combinations configured via options
// can resolve base URLs and API keys before issuing any requests, using the
// shared client.
func Validate(opts ...Option) error {
	return sharedClient().Validate(opts...)
}

// Validate ensures that all model/provider combinations configured via options
// can resolve base URLs and API keys before issuing any requests.
func (c *Client) Validate(opts ...Option) error {
	options := c.callOptions(opts...)

	selector, err := createSelector(options)
	if err != nil {
		return err
	}

	var allErrors []error
	validated := 0
	chain := make(map[string]bool, len(options.LegEfforts))
	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}
		validated++
		chain[legSpecKey(model)] = true
		if err := validateModelConfig(options, model); err != nil {
			allErrors = append(allErrors, err)
		}
	}

	if validated == 0 {
		return fmt.Errorf("no models were configured for validation")
	}

	allErrors = append(allErrors, unmatchedLegEfforts(options, chain)...)

	return errors.Join(allErrors...)
}

// unmatchedLegEfforts reports per-leg overrides naming a spec no leg of the
// chain carries. Without this a typo is silent: the leg it was meant for would
// simply run with the chain-wide effort.
func unmatchedLegEfforts(opts Options, chain map[string]bool) []error {
	unmatched := make([]string, 0, len(opts.LegEfforts))
	for spec := range opts.LegEfforts {
		if !chain[spec] {
			unmatched = append(unmatched, spec)
		}
	}
	// Map order is random and these errors are joined into one message, so sort
	// to keep the same configuration reporting the same text every run.
	slices.Sort(unmatched)

	errs := make([]error, 0, len(unmatched))
	for _, spec := range unmatched {
		errs = append(errs, fmt.Errorf(
			"WithLegReasoningEffort names model %q, which is not in the chain", spec))
	}
	return errs
}

func validateModelConfig(opts Options, model string) error {
	modelSpec := strings.TrimSpace(model)
	prov, modelName, err := parseModelString(modelSpec)
	if err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	if _, err = normalizeReasoningEffort(opts.reasoningEffortFor(modelSpec), prov.Name); err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	base, err := resolveBaseURL(prov, modelName, opts.providerBaseURL(prov.Name))
	if err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	key, err := resolveAPIKey(prov, modelName, opts.providerAPIKey(prov.Name))
	if err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	if err := validateKeyURLPairs(key, base); err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	opts.Logger.Info("validated model configuration", "model", modelSpec, "provider", prov.Name, "base_url", base)
	return nil
}
