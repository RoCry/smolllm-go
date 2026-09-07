package smolllm

import (
	"errors"
	"fmt"
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
	modelSpec := strings.TrimSpace(model)
	prov, modelName, err := parseModelString(modelSpec)
	if err != nil {
		return fmt.Errorf("validate %q: %w", model, err)
	}

	if _, err := normalizeReasoningEffort(opts.ReasoningEffort, prov.Name); err != nil {
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
