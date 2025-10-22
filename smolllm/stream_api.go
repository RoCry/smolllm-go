package smolllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
)

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
	for i, model := range models {
		resp, err := streamOnce(ctx, prompt, options, model)
		if err != nil {
			lastErr = err
			if len(models) > 1 && i < len(models)-1 {
				options.Logger.Warn("model failed, trying fallback", "model", model, "error", err.Error())
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

func streamOnce(ctx context.Context, prompt Prompt, opts Options, model string) (*StreamResponse, error) {
	exec, err := newCallExecution(ctx, prompt, opts, model)
	if err != nil {
		return nil, err
	}

	resp, err := exec.do("starting stream")
	if err != nil {
		exec.cancel()
		return nil, err
	}

	if resp.StatusCode >= http.StatusBadRequest {
		defer resp.Body.Close()
		exec.cancel()
		return nil, httpError(resp)
	}

	chunks, done := startStreamForwarder(opts.Logger, exec.requestContext(), resp, exec.call, exec.cancel, exec.start)

	return &StreamResponse{
		Stream: DeltaStream{
			ch:     chunks,
			done:   done,
			cancel: exec.cancel,
			logger: opts.Logger,
		},
		Model:     exec.call.Model,
		ModelName: exec.call.ModelName,
		Provider:  exec.call.Provider.Name,
	}, nil
}
