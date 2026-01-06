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

	options := applyOptions(opts...)

	selector, err := createSelector(options)
	if err != nil {
		return nil, err
	}

	var lastErr error
	attempted := 0
	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}
		attempted++
		resp, err := streamOnce(ctx, prompt, options, model)
		if err != nil {
			lastErr = err
			options.Logger.Warn("model failed, trying fallback", "model", model, "error", err.Error())
			continue
		}
		return resp, nil
	}

	if lastErr != nil {
		return nil, lastErr
	}

	if attempted == 0 {
		return nil, fmt.Errorf("no models were attempted")
	}
	return nil, fmt.Errorf("all %d models failed", attempted)
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
