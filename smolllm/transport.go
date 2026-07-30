package smolllm

import (
	"bytes"
	"context"
	"encoding/json"
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
	modelSpec, effortOverride := parseModelSpec(model)
	prov, modelName, err := parseModelString(modelSpec)
	if err != nil {
		return nil, err
	}

	base, err := resolveBaseURL(prov, modelName, opts.BaseURL)
	if err != nil {
		return nil, err
	}

	apiKey, err := resolveAPIKey(prov, modelName, opts.APIKey)
	if err != nil {
		return nil, err
	}

	chosenKey, chosenURL, err := balancer.choosePair(apiKey, base)
	if err != nil {
		return nil, err
	}

	reasoningEffort := opts.ReasoningEffort
	if effortOverride != nil {
		reasoningEffort = effortOverride
	}

	url, body, inputTokens, err := buildRequestPayload(
		prompt,
		opts.SystemPrompt,
		modelName,
		prov.Name,
		chosenURL,
		opts.ImagePaths,
		chatPayloadOptions{
			Temperature:        opts.Temperature,
			TopP:               opts.TopP,
			ReasoningEffort:    reasoningEffort,
			MaxTokens:          opts.MaxTokens,
			Stop:               opts.Stop,
			Seed:               opts.Seed,
			IncludeStreamUsage: true,
		},
	)
	if err != nil {
		return nil, err
	}

	return &preparedCall{
		URL:         url,
		Body:        body,
		Provider:    prov,
		Model:       modelSpec,
		ModelName:   modelName,
		APIKey:      chosenKey,
		InputTokens: inputTokens,
	}, nil
}

func resolveBaseURL(prov provider, modelName, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	// Bare model (no provider): explicit option only — never derive env keys
	// from the empty provider name.
	if prov.Name == "" {
		return "", fmt.Errorf(
			"bare model %q requires a base URL. Provide WithBaseURL or use provider/model format", modelName,
		)
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

func resolveAPIKey(prov provider, modelName, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	// Bare model (no provider): explicit option only — never derive env keys
	// from the empty provider name. The ollama literal-key fallback below does
	// not apply either.
	if prov.Name == "" {
		return "", fmt.Errorf(
			"bare model %q requires an API key. Provide WithAPIKey or use provider/model format", modelName,
		)
	}

	envKey := providerEnvKey(prov.Name, "API_KEY")
	value := os.Getenv(envKey)
	if strings.TrimSpace(value) != "" {
		return value, nil
	}

	if prov.Name == providerOllama {
		return providerOllama, nil
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
		start:  time.Time{},
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

func (c *callExecution) retryWithoutStreamUsage(resp *http.Response) (*http.Response, bool, error) {
	if resp.StatusCode != http.StatusBadRequest {
		return resp, false, nil
	}

	body, ok, err := requestBodyWithoutStreamUsage(c.call.Body)
	if err != nil {
		return resp, false, err
	}
	if !ok {
		return resp, false, nil
	}

	originalBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp, false, fmt.Errorf("read original stream_options rejection body: %w", err)
	}
	_ = resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(originalBody))

	req, err := http.NewRequestWithContext(c.req.Context(), http.MethodPost, c.call.URL, bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.call.APIKey)

	c.call.Body = body
	c.call.InputTokens = estimateTokens(string(body))
	c.req = req
	retryResp, err := c.do("retrying request without stream usage")
	if err != nil {
		return nil, false, fmt.Errorf(
			"retry without stream_options failed after original HTTP 400 response %q: %w",
			string(originalBody),
			err,
		)
	}
	return retryResp, true, nil
}

func emitFailureHook(hook func(RequestEvent), call *preparedCall, err error, start time.Time, reported *reportedUsage) {
	if hook == nil || call == nil || err == nil {
		return
	}
	duration := time.Duration(0)
	if !start.IsZero() {
		duration = time.Since(start)
	}
	inputTokens := call.InputTokens
	outputTokens := 0
	estimated := true
	if reported != nil && reported.reported {
		inputTokens = reported.inputTokens
		outputTokens = reported.outputTokens
		estimated = false
	}
	hook(RequestEvent{
		Usage: Usage{
			Provider:     call.Provider.Name,
			Model:        call.Model,
			ModelName:    call.ModelName,
			APIKeyHint:   previewAPIKey(call.APIKey),
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			Duration:     duration,
			TTFT:         0,
			Estimated:    estimated,
		},
		Error:     err,
		Timestamp: time.Now().UTC(),
	})
}

func requestBodyWithoutStreamUsage(body []byte) ([]byte, bool, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, false, fmt.Errorf("decode request body for stream_options retry: %w", err)
	}
	if _, ok := payload["stream_options"]; !ok {
		return body, false, nil
	}
	delete(payload, "stream_options")
	next, err := json.Marshal(payload)
	if err != nil {
		return nil, false, fmt.Errorf("encode request body for stream_options retry: %w", err)
	}
	return next, true, nil
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

// HTTPError represents an HTTP error response from an LLM provider.
type HTTPError struct {
	StatusCode int
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("http error %d: %s", e.StatusCode, e.Body)
}

func httpError(resp *http.Response) error {
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return &HTTPError{StatusCode: resp.StatusCode, Body: fmt.Sprintf("read body: %v", readErr)}
	}
	message := strings.TrimSpace(string(body))
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	return &HTTPError{StatusCode: resp.StatusCode, Body: message}
}
