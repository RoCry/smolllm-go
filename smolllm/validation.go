package smolllm

import (
	"errors"
	"fmt"
)

// Validate ensures that all model/provider combinations configured via options
// can resolve base URLs and API keys before issuing any requests.
func Validate(opts ...Option) error {
	options := applyOptions(opts...)

	selector, err := createSelector(options)
	if err != nil {
		return err
	}

	var allErrors []error
	validated := 0
	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}
		validated++
		if err := validateModelConfig(options, model); err != nil {
			allErrors = append(allErrors, err)
		}
	}

	if validated == 0 {
		return fmt.Errorf("no models were configured for validation")
	}

	return errors.Join(allErrors...)
}

func validateModelConfig(opts Options, model string) error {
	prov, _, err := parseModelString(model)
	if err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	base, err := resolveBaseURL(prov, opts.BaseURL)
	if err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	key, err := resolveAPIKey(prov, opts.APIKey)
	if err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	if err := validateKeyURLPairs(key, base); err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	opts.Logger.Info("validated model configuration", "model", model, "provider", prov.Name, "base_url", base)
	return nil
}
