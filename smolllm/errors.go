package smolllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// HTTPError represents a non-2xx response from an LLM provider.
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

// Disposition says what the fallback chain does about a leg failure.
type Disposition int

const (
	// DispositionRetry re-attempts the same leg after a backoff wait.
	DispositionRetry Disposition = iota
	// DispositionAdvance moves to the next leg of the chain.
	DispositionAdvance
	// DispositionAbort stops the chain immediately without advancing.
	DispositionAbort
)

// String names the disposition for logs and error text.
func (d Disposition) String() string {
	switch d {
	case DispositionRetry:
		return "retry"
	case DispositionAdvance:
		return "advance"
	case DispositionAbort:
		return "abort"
	default:
		return "unknown"
	}
}

// LegError is one leg's failure, carrying the routing identity that failed and
// the policy decision the chain took because of it.
type LegError struct {
	Provider    string
	Model       string
	ModelName   string
	APIKeyHint  string
	Retry       int
	StatusCode  int // 0 when the failure was not an HTTP response
	Disposition Disposition
	Err         error
}

func (e *LegError) Error() string {
	model := e.Model
	if model == "" {
		model = "<unresolved>"
	}
	if e.Retry > 0 {
		return fmt.Sprintf("leg %q (retry %d): %v", model, e.Retry, e.Err)
	}
	return fmt.Sprintf("leg %q: %v", model, e.Err)
}

func (e *LegError) Unwrap() error { return e.Err }

// MarshalJSON renders the classification plus the cause's message. The cause is
// an arbitrary error, which would otherwise serialize to an empty object and
// lose the only part a reader needs.
func (e *LegError) MarshalJSON() ([]byte, error) {
	wire := legErrorWire{
		Provider:    e.Provider,
		Model:       e.Model,
		ModelName:   e.ModelName,
		APIKeyHint:  e.APIKeyHint,
		Retry:       e.Retry,
		StatusCode:  e.StatusCode,
		Disposition: e.Disposition.String(),
		Message:     "",
	}
	if e.Err != nil {
		wire.Message = e.Err.Error()
	}
	encoded, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("encode leg error: %w", err)
	}
	return encoded, nil
}

type legErrorWire struct {
	Provider    string `json:"provider,omitempty"`
	Model       string `json:"model"`
	ModelName   string `json:"model_name,omitempty"`
	APIKeyHint  string `json:"api_key_hint,omitempty"`
	Retry       int    `json:"retry"`
	StatusCode  int    `json:"status_code,omitempty"`
	Disposition string `json:"disposition"`
	Message     string `json:"message"`
}

// newLegError classifies a leg failure and records what failed. call is nil when
// the leg could not even be prepared, which is itself a leg-local problem.
func newLegError(call *preparedCall, model string, retry int, err error) *LegError {
	leg := &LegError{
		Provider:    "",
		Model:       strings.TrimSpace(model),
		ModelName:   "",
		APIKeyHint:  "",
		Retry:       retry,
		StatusCode:  0,
		Disposition: Classify(err),
		Err:         err,
	}
	if call != nil {
		leg.Provider = call.Provider.Name
		leg.Model = call.Model
		leg.ModelName = call.ModelName
		leg.APIKeyHint = previewAPIKey(call.APIKey)
	}
	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		leg.StatusCode = httpErr.StatusCode
	}
	return leg
}

// Classify maps a leg failure to the disposition the chain applies. It is
// exported so a caller can reuse the same policy; the chain applies it already.
//
// The split turns on whether the failure says anything about the *next* leg. A
// malformed request is chain-global and aborts; a credential or quota problem
// belongs to one leg's provider, so the chain advances past it.
func Classify(err error) Disposition {
	if err == nil {
		return DispositionAdvance
	}

	// A cancelled or expired call is the caller's decision, not the provider's.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return DispositionAbort
	}

	var httpErr *HTTPError
	if errors.As(err, &httpErr) {
		switch httpErr.StatusCode {
		case http.StatusBadRequest, http.StatusUnprocessableEntity:
			// The request shape is wrong for every provider, so retrying it
			// anywhere else only burns quota.
			return DispositionAbort
		case http.StatusInternalServerError, http.StatusBadGateway,
			http.StatusServiceUnavailable, http.StatusGatewayTimeout, statusOverloaded:
			return DispositionRetry
		default:
			// 401, 403, 404, 413 and 429 are leg-local: another provider with
			// different credentials, catalogue or capacity may succeed. A 413
			// can describe a provider's request-size or per-request TPM limit.
			return DispositionAdvance
		}
	}

	// Connection, DNS, TLS and EOF failures, plus the post-response guards, all
	// describe this leg only.
	return DispositionAdvance
}

// statusOverloaded is Anthropic's non-standard "overloaded" status.
const statusOverloaded = 529

// asLegError returns err as a LegError, wrapping it when a leg failed before it
// could resolve its own routing identity.
func asLegError(model string, err error) *LegError {
	var leg *LegError
	if errors.As(err, &leg) {
		return leg
	}
	return newLegError(nil, model, 0, err)
}

// joinLegErrors combines every failed leg into one error, so a caller reading
// the message learns why the whole chain failed rather than only its last leg.
func joinLegErrors(legErrors []error) error {
	switch len(legErrors) {
	case 0:
		return errors.New("no models were attempted")
	case 1:
		return legErrors[0]
	default:
		return errors.Join(legErrors...)
	}
}
