package smolllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

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
	exec, err := newCallExecution(ctx, prompt, opts, model)
	if err != nil {
		return nil, err
	}
	defer exec.cancel()

	resp, err := exec.do("sending request")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		return nil, httpError(resp)
	}

	result, ttft, err := consumeStream(opts.Logger, exec.requestContext(), resp.Body, opts.StreamHandler, exec.start)
	if err != nil {
		return nil, err
	}

	if opts.RemoveBackticks {
		result = stripBackticks(result)
	}
	if strings.TrimSpace(result) == "" {
		return nil, fmt.Errorf("model %q returned empty response", exec.call.Model)
	}

	total := time.Since(exec.start)
	outputTokens := estimateTokens(result)
	opts.Logger.Info(
		formatMetrics(exec.call.ModelName, exec.call.InputTokens, outputTokens, total, ttft),
		"model", exec.call.ModelName,
	)

	return &Response{
		Text:      result,
		Model:     exec.call.Model,
		ModelName: exec.call.ModelName,
		Provider:  exec.call.Provider.Name,
	}, nil
}
