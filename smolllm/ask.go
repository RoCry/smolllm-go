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

	options := applyOptions(opts...)

	selector, err := createSelector(options)
	if err != nil {
		return nil, err
	}

	var lastErr error
	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}
		resp, err := withRetry(ctx, options.Logger, model, func() (*Response, error) {
			return askOnce(ctx, prompt, options, model)
		})
		if err != nil {
			lastErr = err
			if selector.HasMore() {
				options.Logger.Warn("model failed, trying fallback", "model", model, "error", err.Error())
			} else {
				options.Logger.Warn("model failed", "model", model, "error", err.Error())
			}
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

	cr, err := consumeStream(opts.Logger, exec.requestContext(), resp.Body, opts.StreamHandler, exec.start)
	if err != nil {
		return nil, err
	}

	content := cr.content
	if opts.RemoveBackticks {
		content = stripBackticks(content)
	}
	if strings.TrimSpace(content) == "" && strings.TrimSpace(cr.reasoning) == "" {
		return nil, fmt.Errorf("model %q returned empty response", exec.call.Model)
	}

	total := time.Since(exec.start)
	outputTokens := estimateTokens(content + cr.reasoning)
	opts.Logger.Info(
		formatMetrics(exec.call.ModelName, exec.call.InputTokens, outputTokens, total, cr.ttft),
		"model", exec.call.ModelName,
	)

	usage := Usage{
		Provider:     exec.call.Provider.Name,
		Model:        exec.call.Model,
		ModelName:    exec.call.ModelName,
		APIKeyHint:   previewAPIKey(exec.call.APIKey),
		InputTokens:  exec.call.InputTokens,
		OutputTokens: outputTokens,
		Duration:     total,
		TTFT:         cr.ttft,
	}
	if opts.Hook != nil {
		opts.Hook(RequestEvent{Usage: usage, Timestamp: time.Now().UTC()})
	}

	return &Response{
		Text:      content,
		Reasoning: cr.reasoning,
		Model:     exec.call.Model,
		ModelName: exec.call.ModelName,
		Provider:  exec.call.Provider.Name,
		Usage:     usage,
	}, nil
}
