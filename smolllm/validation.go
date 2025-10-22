package smolllm

import "fmt"

// Validate ensures that all model/provider combinations configured via options
// can resolve base URLs and API keys before issuing any requests.
func Validate(opts ...Option) error {
	options := applyOptions(opts...)

	models, err := resolveModels(options.Model)
	if err != nil {
		return err
	}

	for _, model := range models {
		if err := validateModelConfig(options, model); err != nil {
			return err
		}
	}

	return nil
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

	opts.Logger.Info("validated model configuration", "model", model, "provider", prov.Name)
	return nil
}
