package smolllm

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"
)

type preparedCall struct {
	URL         string
	Body        []byte
	Provider    provider
	Model       string
	ModelName   string
	APIKey      string
	InputTokens int
}

func prepareLLMCall(prompt Prompt, opts Options, model string) (*preparedCall, error) {
	prov, modelName, err := parseModelString(model)
	if err != nil {
		return nil, err
	}

	base, err := resolveBaseURL(prov, opts.BaseURL)
	if err != nil {
		return nil, err
	}

	apiKey, err := resolveAPIKey(prov, opts.APIKey)
	if err != nil {
		return nil, err
	}

	chosenKey, chosenURL, err := balancer.choosePair(apiKey, base)
	if err != nil {
		return nil, err
	}

	url, body, inputTokens, err := buildRequestPayload(prompt, opts.SystemPrompt, modelName, prov.Name, chosenURL, opts.ImagePaths)
	if err != nil {
		return nil, err
	}

	return &preparedCall{
		URL:         url,
		Body:        body,
		Provider:    prov,
		Model:       model,
		ModelName:   modelName,
		APIKey:      chosenKey,
		InputTokens: inputTokens,
	}, nil
}

func resolveModels(explicit string) ([]string, error) {
	candidate := strings.TrimSpace(explicit)
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

	return models, nil
}

func resolveBaseURL(prov provider, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	envKey := providerEnvKey(prov.Name, "BASE_URL")
	value := os.Getenv(envKey)
	if strings.TrimSpace(value) != "" {
		return value, nil
	}

	if strings.TrimSpace(prov.BaseURL) == "" {
		return "", fmt.Errorf("base URL not found. set %s or provide WithBaseURL", envKey)
	}
	return prov.BaseURL, nil
}

func resolveAPIKey(prov provider, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	envKey := providerEnvKey(prov.Name, "API_KEY")
	value := os.Getenv(envKey)
	if strings.TrimSpace(value) != "" {
		return value, nil
	}

	if prov.Name == "ollama" {
		return "ollama", nil
	}
	return "", fmt.Errorf("API key not found. set %s or provide WithAPIKey", envKey)
}

func providerEnvKey(providerName, suffix string) string {
	base := strings.ToUpper(strings.ReplaceAll(providerName, "-", "_"))
	return base + "_" + suffix
}

type callExecution struct {
	call   *preparedCall
	client *http.Client
	req    *http.Request
	cancel context.CancelFunc
	start  time.Time
	logger *slog.Logger
}

func newCallExecution(ctx context.Context, prompt Prompt, opts Options, model string) (*callExecution, error) {
	call, err := prepareLLMCall(prompt, opts, model)
	if err != nil {
		return nil, err
	}

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	reqCtx, cancel := deriveContext(ctx, opts.Timeout)

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, call.URL, bytes.NewReader(call.Body))
	if err != nil {
		cancel()
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+call.APIKey)

	return &callExecution{
		call:   call,
		client: client,
		req:    req,
		cancel: cancel,
		logger: opts.Logger,
	}, nil
}

func (c *callExecution) do(event string) (*http.Response, error) {
	c.logger.Info(
		event,
		"url", c.call.URL,
		"model", c.call.ModelName,
		"api_key", previewAPIKey(c.call.APIKey),
		"approx_tokens", c.call.InputTokens,
	)

	c.start = time.Now().UTC()
	resp, err := c.client.Do(c.req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (c *callExecution) requestContext() context.Context {
	return c.req.Context()
}

func deriveContext(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func previewAPIKey(key string) string {
	if len(key) <= 9 {
		return key
	}
	return key[:5] + "..." + key[len(key)-4:]
}

func httpError(resp *http.Response) error {
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return fmt.Errorf("http error %d: read body: %w", resp.StatusCode, readErr)
	}
	message := strings.TrimSpace(string(body))
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	return fmt.Errorf("http error %d: %s", resp.StatusCode, message)
}
