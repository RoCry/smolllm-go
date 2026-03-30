package smolllm

import (
	"context"
	"errors"
	"log/slog"
	"time"
)

const (
	defaultMaxRetries = 3
	retryBaseDelay    = 2 * time.Second
	retryMaxDelay     = 30 * time.Second
	retryBackoffScale = 3
)

func isRetryableHTTPStatus(code int) bool {
	switch code {
	case 429, 500, 502, 503, 529:
		return true
	}
	return false
}

func isRetryableError(err error) bool {
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		return isRetryableHTTPStatus(httpErr.StatusCode)
	}
	return false
}

func retryDelay(attempt int) time.Duration {
	d := retryBaseDelay
	for range attempt {
		d *= retryBackoffScale
	}
	if d > retryMaxDelay {
		d = retryMaxDelay
	}
	return d
}

// withRetry runs fn up to maxRetries times, retrying only on transient HTTP errors.
func withRetry[T any](ctx context.Context, logger *slog.Logger, model string, fn func() (T, error)) (T, error) {
	var zero T
	var lastErr error
	for attempt := range defaultMaxRetries {
		if attempt > 0 {
			delay := retryDelay(attempt)
			logger.Warn("retrying after transient error", "model", model, "attempt", attempt+1, "delay", delay, "err", lastErr)
			t := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				t.Stop()
				return zero, lastErr
			case <-t.C:
			}
		}
		result, err := fn()
		if err == nil {
			return result, nil
		}
		lastErr = err
		if !isRetryableError(err) {
			return zero, err
		}
	}
	return zero, lastErr
}
