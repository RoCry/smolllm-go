package smolllm

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClassify(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
		want Disposition
	}{
		// A malformed request is wrong for every provider in the chain.
		{"400 bad request", &HTTPError{StatusCode: 400, Body: "bad request"}, DispositionAbort},
		{"413 payload too large", &HTTPError{StatusCode: 413, Body: "too large"}, DispositionAbort},
		{"422 unprocessable", &HTTPError{StatusCode: 422, Body: "unprocessable"}, DispositionAbort},

		// Credentials and catalogue belong to one leg's provider.
		{"401 unauthorized", &HTTPError{StatusCode: 401, Body: "unauthorized"}, DispositionAdvance},
		{"403 forbidden", &HTTPError{StatusCode: 403, Body: "forbidden"}, DispositionAdvance},
		{"404 not found", &HTTPError{StatusCode: 404, Body: "not found"}, DispositionAdvance},
		{"429 rate limited", &HTTPError{StatusCode: 429, Body: "slow down"}, DispositionAdvance},

		// Transient server faults deserve the same leg again.
		{"500 internal", &HTTPError{StatusCode: 500, Body: "internal"}, DispositionRetry},
		{"502 bad gateway", &HTTPError{StatusCode: 502, Body: "bad gateway"}, DispositionRetry},
		{"503 unavailable", &HTTPError{StatusCode: 503, Body: "unavailable"}, DispositionRetry},
		{"504 gateway timeout", &HTTPError{StatusCode: 504, Body: "timeout"}, DispositionRetry},
		{"529 overloaded", &HTTPError{StatusCode: 529, Body: "overloaded"}, DispositionRetry},

		// Transport faults describe this leg's endpoint only.
		{"connection refused", errors.New("connection refused"), DispositionAdvance},
		{"dns failure", dnsFailure(), DispositionAdvance},
		{"unexpected eof", io.ErrUnexpectedEOF, DispositionAdvance},

		// The caller's own decision stops the chain.
		{"context canceled", context.Canceled, DispositionAbort},
		{"deadline exceeded", context.DeadlineExceeded, DispositionAbort},

		// Post-response guards fail one leg and route around it.
		{"empty response guard", errors.New(`model "x" returned empty response`), DispositionAdvance},
		{"truncation guard", errors.New(`model "x" truncated before any content`), DispositionAdvance},

		{"nil error", nil, DispositionAdvance},

		// Classification sees through wrapping, including a LegError.
		{"wrapped 500", fmt.Errorf("wrapped: %w", &HTTPError{StatusCode: 500, Body: "x"}), DispositionRetry},
		{
			"leg error wrapping a 400",
			&LegError{
				Provider: "openai", Model: "openai/gpt-5", ModelName: "gpt-5", APIKeyHint: "k",
				Retry: 0, StatusCode: 400, Disposition: DispositionAbort,
				Err: &HTTPError{StatusCode: 400, Body: "bad request"},
			},
			DispositionAbort,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, Classify(tt.err), "disposition for %v", tt.err)
		})
	}
}

// dnsFailure builds a name-resolution error, which is what an unreachable
// provider endpoint looks like to the chain.
func dnsFailure() error {
	return &net.DNSError{ //nolint:exhaustruct // only the fields Classify can see matter here
		Err:  "no such host",
		Name: "x.example",
	}
}

func TestDispositionString(t *testing.T) {
	t.Parallel()
	assert.Equal(t, "retry", DispositionRetry.String())
	assert.Equal(t, "advance", DispositionAdvance.String())
	assert.Equal(t, "abort", DispositionAbort.String())
}

func TestLegErrorUnwrapsToItsCause(t *testing.T) {
	t.Parallel()

	cause := &HTTPError{StatusCode: 503, Body: "unavailable"}
	leg := newLegError(nil, "openai/gpt-5", 2, cause)

	assert.Equal(t, "openai/gpt-5", leg.Model)
	assert.Equal(t, 503, leg.StatusCode)
	assert.Equal(t, 2, leg.Retry)
	assert.Equal(t, DispositionRetry, leg.Disposition)
	require.ErrorIs(t, leg, cause)
	assert.Contains(t, leg.Error(), "retry 2")
	assert.Contains(t, leg.Error(), "unavailable")
}
