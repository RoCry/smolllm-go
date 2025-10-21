package smolllm

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
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

// Ask performs a single prompt/response exchange, streaming under the hood before returning the full text.
func Ask(ctx context.Context, prompt Prompt, opts ...Option) (*Response, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}
	options := defaultOptions()
	for _, opt := range opts {
		opt(&options)
	}

	models, err := resolveModels(options.Model)
	if err != nil {
		return nil, err
	}

	var lastErr error
	for _, model := range models {
		resp, err := askOnce(ctx, prompt, options, model)
		if err != nil {
			lastErr = err
			logger.Warn("ask failed", "model", model, "err", err)
			continue
		}
		return resp, nil
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, fmt.Errorf("no models were attempted")
}

// Stream returns a streaming response channel for the given prompt.
func Stream(ctx context.Context, prompt Prompt, opts ...Option) (*StreamResponse, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}
	options := defaultOptions()
	for _, opt := range opts {
		opt(&options)
	}

	models, err := resolveModels(options.Model)
	if err != nil {
		return nil, err
	}

	var lastErr error
	for _, model := range models {
		resp, err := streamOnce(ctx, prompt, options, model)
		if err != nil {
			lastErr = err
			logger.Warn("stream failed", "model", model, "err", err)
			continue
		}
		return resp, nil
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, fmt.Errorf("no models were attempted")
}

func askOnce(ctx context.Context, prompt Prompt, opts Options, model string) (*Response, error) {
	call, err := prepareLLMCall(prompt, opts, model)
	if err != nil {
		return nil, err
	}

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	reqCtx, cancel := deriveContext(ctx, opts.Timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, call.URL, bytes.NewReader(call.Body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+call.APIKey)

	logger.Info(
		"sending request",
		"url", call.URL,
		"model", call.ModelName,
		"api_key", previewAPIKey(call.APIKey),
		"approx_tokens", call.InputTokens,
	)

	start := time.Now().UTC()
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("http error %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	result, ttft, err := consumeStream(reqCtx, resp.Body, opts.StreamHandler, start)
	if err != nil {
		return nil, err
	}

	if opts.RemoveBackticks {
		result = stripBackticks(result)
	}
	if strings.TrimSpace(result) == "" {
		return nil, fmt.Errorf("model %q returned empty response", call.Model)
	}

	total := time.Since(start)
	outputTokens := estimateTokens(result)
	logger.Info(
		formatMetrics(call.ModelName, call.InputTokens, outputTokens, total, ttft),
		"model", call.ModelName,
	)

	return &Response{
		Text:      result,
		Model:     call.Model,
		ModelName: call.ModelName,
		Provider:  call.Provider.Name,
	}, nil
}

func streamOnce(ctx context.Context, prompt Prompt, opts Options, model string) (*StreamResponse, error) {
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

	logger.Info(
		"starting stream",
		"url", call.URL,
		"model", call.ModelName,
		"api_key", previewAPIKey(call.APIKey),
		"approx_tokens", call.InputTokens,
	)

	start := time.Now().UTC()
	resp, err := client.Do(req)
	if err != nil {
		cancel()
		return nil, err
	}

	if resp.StatusCode >= http.StatusBadRequest {
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		cancel()
		return nil, fmt.Errorf("http error %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	chunks := make(chan string)
	done := make(chan error, 1)

	go func() {
		defer cancel()
		defer resp.Body.Close()
		defer close(chunks)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

		var firstToken time.Time
		var err error
		var builder strings.Builder

		for scanner.Scan() {
			select {
			case <-reqCtx.Done():
				err = reqCtx.Err()
				break
			default:
			}

			line := scanner.Text()
			var delta string
			delta, err = processChunkLine(line)
			if err != nil {
				break
			}
			if delta == "" {
				continue
			}
			if firstToken.IsZero() {
				firstToken = time.Now().UTC()
			}
			builder.WriteString(delta)
			select {
			case <-reqCtx.Done():
				err = reqCtx.Err()
				break
			case chunks <- delta:
			}
		}

		if err == nil {
			err = scanner.Err()
		}

		total := time.Since(start)
		var ttft time.Duration = -1
		if !firstToken.IsZero() {
			ttft = firstToken.Sub(start)
			if ttft < 0 {
				ttft = 0
			}
		}

		if err != nil && !errors.Is(err, context.Canceled) {
			logger.Warn("stream terminated with error", "model", call.ModelName, "err", err)
		} else {
			outputTokens := estimateTokens(builder.String())
			logger.Info(
				formatMetrics(call.ModelName, call.InputTokens, outputTokens, total, ttft),
				"model", call.ModelName,
			)
		}

		done <- err
		close(done)
	}()

	return &StreamResponse{
		Stream: DeltaStream{
			ch:     chunks,
			done:   done,
			cancel: cancel,
		},
		Model:     call.Model,
		ModelName: call.ModelName,
		Provider:  call.Provider.Name,
	}, nil
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

func deriveContext(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func consumeStream(ctx context.Context, reader io.Reader, handler func(context.Context, string) error, start time.Time) (string, time.Duration, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var builder strings.Builder
	var firstToken time.Time

	for scanner.Scan() {
		if ctx.Err() != nil {
			return "", -1, ctx.Err()
		}

		line := scanner.Text()
		delta, err := processChunkLine(line)
		if err != nil {
			return "", -1, err
		}
		if delta == "" {
			continue
		}
		if firstToken.IsZero() {
			firstToken = time.Now().UTC()
		}
		if handler != nil {
			if err := handler(ctx, delta); err != nil {
				return "", -1, err
			}
		}
		builder.WriteString(delta)
	}

	if err := scanner.Err(); err != nil {
		return "", -1, err
	}

	result := strings.TrimSpace(builder.String())
	if result == "" {
		return "", -1, fmt.Errorf("empty response from model")
	}

	var ttft time.Duration = -1
	if !firstToken.IsZero() {
		ttft = firstToken.Sub(start)
		if ttft < 0 {
			ttft = 0
		}
	}

	return result, ttft, nil
}

func previewAPIKey(key string) string {
	if len(key) <= 9 {
		return key
	}
	return key[:5] + "..." + key[len(key)-4:]
}
