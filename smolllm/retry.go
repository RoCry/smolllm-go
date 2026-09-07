package smolllm

import (
	"context"
	"log/slog"
	"time"
)

const (
	defaultMaxRetries = 3
	retryBaseDelay    = 1 * time.Second
	retryMaxDelay     = 30 * time.Second
	retryBackoffScale = 2
)

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

// withRetry runs fn up to maxRetries times, retrying only on transient HTTP
// errors. The backoff waits share the caller's context, so the whole-call
// deadline bounds them too.
func withRetry[T any](
	ctx context.Context, logger *slog.Logger, model string, maxRetries int, fn func(retry int) (T, error),
) (T, error) {
	var zero T
	var lastErr error
	for attempt := range maxRetries {
		if attempt > 0 {
			// Delays: attempt=1 -> 1s, attempt=2 -> 2s, attempt=3 -> 4s, ...
			delay := retryDelay(attempt - 1)
			logger.Warn("retrying after transient error", "model", model, "attempt", attempt+1, "delay", delay, "err", lastErr)
			t := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				t.Stop()
				return zero, lastErr
			case <-t.C:
			}
		}
		result, err := fn(attempt)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if Classify(err) != DispositionRetry {
			return zero, err
		}
	}
	return zero, lastErr
}
