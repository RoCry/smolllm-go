package smolllm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"testing"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
}

func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		err       error
		retryable bool
	}{
		{&HTTPError{StatusCode: 429, Body: "rate limited"}, true},
		{&HTTPError{StatusCode: 500, Body: "internal"}, true},
		{&HTTPError{StatusCode: 502, Body: "bad gateway"}, true},
		{&HTTPError{StatusCode: 503, Body: "unavailable"}, true},
		{&HTTPError{StatusCode: 529, Body: "overloaded"}, true},
		{&HTTPError{StatusCode: 400, Body: "bad request"}, false},
		{&HTTPError{StatusCode: 401, Body: "unauthorized"}, false},
		{&HTTPError{StatusCode: 404, Body: "not found"}, false},
		{errors.New("connection refused"), false},
		{fmt.Errorf("wrapped: %w", &HTTPError{StatusCode: 429, Body: "nested"}), true},
	}

	for _, tt := range tests {
		got := isRetryableError(tt.err)
		if got != tt.retryable {
			t.Errorf("isRetryableError(%v) = %v, want %v", tt.err, got, tt.retryable)
		}
	}
}

func TestWithRetry_SucceedsFirstAttempt(t *testing.T) {
	calls := 0
	result, err := withRetry(context.Background(), testLogger(), "test-model", func() (string, error) {
		calls++
		return "ok", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != "ok" {
		t.Fatalf("got %q, want %q", result, "ok")
	}
	if calls != 1 {
		t.Fatalf("expected 1 call, got %d", calls)
	}
}

func TestWithRetry_RetriesOnTransient(t *testing.T) {
	calls := 0
	result, err := withRetry(context.Background(), testLogger(), "test-model", func() (string, error) {
		calls++
		if calls < 3 {
			return "", &HTTPError{StatusCode: 429, Body: "rate limited"}
		}
		return "recovered", nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result != "recovered" {
		t.Fatalf("got %q, want %q", result, "recovered")
	}
	if calls != 3 {
		t.Fatalf("expected 3 calls, got %d", calls)
	}
}

func TestWithRetry_NonRetryableFails(t *testing.T) {
	calls := 0
	_, err := withRetry(context.Background(), testLogger(), "test-model", func() (string, error) {
		calls++
		return "", &HTTPError{StatusCode: 401, Body: "unauthorized"}
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Fatalf("expected 1 call (no retry for 401), got %d", calls)
	}
}

func TestWithRetry_ExhaustsRetries(t *testing.T) {
	calls := 0
	_, err := withRetry(context.Background(), testLogger(), "test-model", func() (string, error) {
		calls++
		return "", &HTTPError{StatusCode: 429, Body: "rate limited"}
	})
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if calls != defaultMaxRetries {
		t.Fatalf("expected %d calls, got %d", defaultMaxRetries, calls)
	}
}

func TestWithRetry_CancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	calls := 0
	_, err := withRetry(ctx, testLogger(), "test-model", func() (string, error) {
		calls++
		cancel()
		return "", &HTTPError{StatusCode: 429, Body: "rate limited"}
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if calls != 1 {
		t.Fatalf("expected 1 call before cancel, got %d", calls)
	}
}

func TestRetryDelay(t *testing.T) {
	d0 := retryDelay(0)
	d1 := retryDelay(1)
	d2 := retryDelay(2)

	if d0 >= d1 || d1 >= d2 {
		t.Fatalf("delays should increase: %v, %v, %v", d0, d1, d2)
	}

	dHuge := retryDelay(100)
	if dHuge > retryMaxDelay {
		t.Fatalf("delay %v exceeds max %v", dHuge, retryMaxDelay)
	}
}

func TestHTTPErrorFormat(t *testing.T) {
	err := &HTTPError{StatusCode: 429, Body: "too many requests"}
	want := "http error 429: too many requests"
	if err.Error() != want {
		t.Fatalf("got %q, want %q", err.Error(), want)
	}
}
