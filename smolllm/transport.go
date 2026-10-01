package smolllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
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
	Headers     map[string]string
	InputTokens int
}

// applyHeaders sets the standard headers, then the caller's own so they win.
func (p *preparedCall) applyHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.APIKey)
	for name, value := range p.Headers {
		req.Header.Set(name, value)
	}
}

func prepareLLMCall(turn Request, opts Options, model string, bal *simpleBalancer) (*preparedCall, error) {
	modelSpec := strings.TrimSpace(model)
	prov, modelName, err := parseModelString(modelSpec)
	if err != nil {
		return nil, err
	}

	base, err := resolveBaseURL(prov, modelName, opts.providerBaseURL(prov.Name))
	if err != nil {
		return nil, err
	}

	apiKey, err := resolveAPIKey(prov, modelName, opts.providerAPIKey(prov.Name))
	if err != nil {
		return nil, err
	}

	chosenKey, chosenURL, err := bal.choosePair(apiKey, base)
	if err != nil {
		return nil, err
	}

	url, body, inputTokens, err := buildRequestPayload(
		turn,
		modelName,
		prov.Name,
		chosenURL,
		opts.ImagePaths,
		chatPayloadOptions{
			Temperature:        opts.Temperature,
			TopP:               opts.TopP,
			ReasoningEffort:    opts.reasoningEffortFor(modelSpec),
			MaxTokens:          opts.MaxTokens,
			Stop:               opts.Stop,
			Seed:               opts.Seed,
			IncludeStreamUsage: true,
			ExtraBody:          opts.extraBodyFor(prov.Name),
		},
	)
	if err != nil {
		return nil, err
	}

	return &preparedCall{
		URL:         url,
		Body:        body,
		Provider:    prov,
		Model:       modelSpec,
		ModelName:   modelName,
		APIKey:      chosenKey,
		Headers:     opts.providerHeaders(prov.Name),
		InputTokens: inputTokens,
	}, nil
}

func resolveBaseURL(prov provider, modelName, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	// Bare model (no provider): explicit option only — never derive env keys
	// from the empty provider name.
	if prov.Name == BareProvider {
		return "", fmt.Errorf(
			"bare model %q requires a base URL. Provide WithProvider(BareProvider, ...) or use provider/model format",
			modelName,
		)
	}

	envKey := providerEnvKey(prov.Name, "BASE_URL")
	value := os.Getenv(envKey)
	if strings.TrimSpace(value) != "" {
		return value, nil
	}

	if strings.TrimSpace(prov.BaseURL) == "" {
		return "", fmt.Errorf("base URL not found. set %s or provide WithProvider", envKey)
	}
	return prov.BaseURL, nil
}

func resolveAPIKey(prov provider, modelName, explicit string) (string, error) {
	if strings.TrimSpace(explicit) != "" {
		return explicit, nil
	}

	// Bare model (no provider): explicit option only — never derive env keys
	// from the empty provider name. The ollama literal-key fallback below does
	// not apply either.
	if prov.Name == BareProvider {
		return "", fmt.Errorf(
			"bare model %q requires an API key. Provide WithProvider(BareProvider, ...) or use provider/model format",
			modelName,
		)
	}

	envKey := providerEnvKey(prov.Name, "API_KEY")
	value := os.Getenv(envKey)
	if strings.TrimSpace(value) != "" {
		return value, nil
	}

	if prov.Name == providerOllama {
		return providerOllama, nil
	}
	return "", fmt.Errorf("API key not found. set %s or provide WithProvider", envKey)
}

func providerEnvKey(providerName, suffix string) string {
	base := strings.ToUpper(strings.ReplaceAll(providerName, "-", "_"))
	return base + "_" + suffix
}

type callExecution struct {
	call   *preparedCall
	client *http.Client
	req    *http.Request
	cancel context.CancelCauseFunc
	start  time.Time
	logger *slog.Logger
}

func newCallExecution(
	ctx context.Context, turn Request, opts Options, model string, bal *simpleBalancer,
) (*callExecution, error) {
	call, err := prepareLLMCall(turn, opts, model, bal)
	if err != nil {
		return nil, err
	}

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	// The whole-call deadline already lives on ctx; this cancel tears down the
	// connection once the attempt is done, or early when the leg budget runs out.
	reqCtx, cancel := context.WithCancelCause(ctx)

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, call.URL, bytes.NewReader(call.Body))
	if err != nil {
		cancel(nil)
		return nil, err
	}
	call.applyHeaders(req)

	return &callExecution{
		call:   call,
		client: client,
		req:    req,
		cancel: cancel,
		start:  time.Time{},
		logger: opts.Logger,
	}, nil
}

func (c *callExecution) do(event string) (*http.Response, error) {
	c.logger.Info(
		event,
		"url", c.call.URL,
		"model", c.call.ModelName,
		"api_key", previewAPIKey(c.call.APIKey),
		"approx_tokens", c.call.InputTokens,
	)

	c.start = time.Now().UTC()
	resp, err := c.client.Do(c.req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func (c *callExecution) retryWithoutStreamUsage(resp *http.Response) (*http.Response, bool, error) {
	if resp.StatusCode != http.StatusBadRequest {
		return resp, false, nil
	}

	body, ok, err := requestBodyWithoutStreamUsage(c.call.Body)
	if err != nil {
		return resp, false, err
	}
	if !ok {
		return resp, false, nil
	}

	originalBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp, false, fmt.Errorf("read original stream_options rejection body: %w", err)
	}
	resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(originalBody))

	req, err := http.NewRequestWithContext(c.req.Context(), http.MethodPost, c.call.URL, bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	c.call.applyHeaders(req)

	c.call.Body = body
	c.call.InputTokens = estimateTokens(string(body))
	c.req = req
	retryResp, err := c.do("retrying request without stream usage")
	if err != nil {
		return nil, false, fmt.Errorf(
			"retry without stream_options failed after original HTTP 400 response %q: %w",
			string(originalBody),
			err,
		)
	}
	return retryResp, true, nil
}

// failedAttempt describes a leg that did not produce a usable response. outcome
// is nil when the leg failed before its stream was read. When the provider still
// reported usage before failing, those counts are kept: the tokens were spent
// either way.
func failedAttempt(
	call *preparedCall, retry int, leg *LegError, start time.Time, outcome *legOutcome,
) Attempt {
	attempt := newAttempt(call, retry)
	if !start.IsZero() {
		attempt.Duration = time.Since(start)
	}
	if outcome != nil {
		attempt.TTFT = outcome.ttft
		if outcome.usage.reported {
			attempt.Usage = outcome.usage.usage
		}
	}
	attempt.Err = leg
	return attempt
}

func requestBodyWithoutStreamUsage(body []byte) ([]byte, bool, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, false, fmt.Errorf("decode request body for stream_options retry: %w", err)
	}
	if _, ok := payload["stream_options"]; !ok {
		return body, false, nil
	}
	delete(payload, "stream_options")
	next, err := json.Marshal(payload)
	if err != nil {
		return nil, false, fmt.Errorf("encode request body for stream_options retry: %w", err)
	}
	return next, true, nil
}

func (c *callExecution) requestContext() context.Context {
	return c.req.Context()
}

// ErrLegBudgetExceeded marks a leg that ran out of its WithLegBudget allowance
// before its response started. It says nothing about the next leg, so the chain
// advances.
var ErrLegBudgetExceeded = errors.New("leg budget exceeded before the response started")

// armLegBudget cancels the attempt once deadline passes, unless the returned
// disarm runs first. A zero deadline arms nothing. disarm reports whether the
// budget had already fired.
func (c *callExecution) armLegBudget(deadline time.Time) (disarm func() bool) {
	if deadline.IsZero() {
		return func() bool { return false }
	}
	timer := time.AfterFunc(time.Until(deadline), func() { c.cancel(ErrLegBudgetExceeded) })
	return func() bool { return !timer.Stop() }
}

// budgetCause rewrites err as a leg-budget failure when that is why the
// attempt was cancelled; a caller cancellation or the whole-call deadline keeps
// its own error, since those end the chain rather than the leg.
func (c *callExecution) budgetCause(err error, budget time.Duration) error {
	if errors.Is(context.Cause(c.req.Context()), ErrLegBudgetExceeded) {
		return fmt.Errorf("%w (%s)", ErrLegBudgetExceeded, budget)
	}
	return err
}

func deriveContext(ctx context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

func previewAPIKey(key string) string {
	if len(key) <= 9 {
		return key
	}
	return key[:5] + "..." + key[len(key)-4:]
}
