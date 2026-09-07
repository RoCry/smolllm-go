package smolllm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// EmbeddingResponse holds the result of an Embed call.
type EmbeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
	Model      string      `json:"model"`
	ModelName  string      `json:"model_name"`
	Provider   string      `json:"provider"`
	Usage      Usage       `json:"usage"`
}

type embeddingRequest struct {
	Model           string  `json:"model"`
	Input           any     `json:"input"`
	Dimensions      int     `json:"dimensions,omitempty"`
	ReasoningEffort *string `json:"reasoning_effort,omitempty"`
}

type embeddingDataItem struct {
	Index     int       `json:"index"`
	Embedding []float64 `json:"embedding"`
}

type embeddingAPIResponse struct {
	Data  []embeddingDataItem `json:"data"`
	Model string              `json:"model"`
	Usage *struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// Embed generates embeddings for the given input strings using the shared
// client and an OpenAI-compatible endpoint.
func Embed(ctx context.Context, input []string, opts ...Option) (*EmbeddingResponse, error) {
	return sharedClient().Embed(ctx, input, opts...)
}

// Embed generates embeddings for the given input strings using an
// OpenAI-compatible endpoint.
func (c *Client) Embed(ctx context.Context, input []string, opts ...Option) (*EmbeddingResponse, error) {
	if ctx == nil {
		return nil, errors.New("context must not be nil")
	}
	if len(input) == 0 {
		return nil, errors.New("input must not be empty")
	}

	options := c.callOptions(opts...)

	selector, err := createSelector(options)
	if err != nil {
		return nil, err
	}

	// One deadline bounds the whole call: every leg, every retry and the backoff
	// waits between them.
	callCtx, cancelCall := deriveContext(ctx, options.Timeout)
	defer cancelCall()

	var legErrors []error
	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}
		resp, err := withRetry(callCtx, options.Logger, model, options.MaxRetries,
			func(retry int) (*EmbeddingResponse, error) {
				return c.embedOnce(callCtx, input, options, model, retry)
			})
		if err != nil {
			leg := asLegError(model, err)
			legErrors = append(legErrors, leg)
			options.Logger.Warn("leg failed",
				"model", model,
				"disposition", leg.Disposition.String(),
				"error", err.Error(),
			)
			if leg.Disposition == DispositionAbort {
				break
			}
			continue
		}
		return resp, nil
	}

	return nil, joinLegErrors(legErrors)
}

func (c *Client) embedOnce(
	ctx context.Context, input []string, opts Options, model string, retry int,
) (*EmbeddingResponse, error) {
	modelSpec := strings.TrimSpace(model)
	prov, modelName, err := parseModelString(modelSpec)
	if err != nil {
		return nil, newLegError(nil, model, retry, err)
	}

	base, err := resolveBaseURL(prov, modelName, opts.providerBaseURL(prov.Name))
	if err != nil {
		return nil, newLegError(nil, model, retry, err)
	}

	apiKey, err := resolveAPIKey(prov, modelName, opts.providerAPIKey(prov.Name))
	if err != nil {
		return nil, newLegError(nil, model, retry, err)
	}

	chosenKey, chosenURL, err := c.balancer.choosePair(apiKey, base)
	if err != nil {
		return nil, newLegError(nil, model, retry, err)
	}

	url := resolveEndpointURL(chosenURL, prov.Name, "embeddings")

	// Single string input for len==1, slice otherwise.
	var inputPayload any
	if len(input) == 1 {
		inputPayload = input[0]
	} else {
		inputPayload = input
	}

	normalizedReasoningEffort, err := normalizeReasoningEffort(opts.reasoningEffortFor(modelSpec), prov.Name)
	if err != nil {
		return nil, newLegError(nil, model, retry, err)
	}

	body, err := json.Marshal(embeddingRequest{
		Model:           modelName,
		Input:           inputPayload,
		Dimensions:      opts.Dimensions,
		ReasoningEffort: normalizedReasoningEffort,
	})
	if err != nil {
		return nil, fmt.Errorf("encode embedding request: %w", err)
	}
	inputTokens := estimateTokens(string(body))
	// Embed has no preparedCall, so it describes its own leg identity.
	identity := Attempt{
		Provider:   prov.Name,
		Model:      modelSpec,
		ModelName:  modelName,
		APIKeyHint: previewAPIKey(chosenKey),
		Retry:      retry,
		Usage:      newUsage(inputTokens, 0, 0, 0, true),
		Duration:   0,
		TTFT:       0,
		Err:        nil,
	}
	fail := func(err error, start time.Time) (*EmbeddingResponse, error) {
		leg := &LegError{
			Provider:    prov.Name,
			Model:       modelSpec,
			ModelName:   modelName,
			APIKeyHint:  previewAPIKey(chosenKey),
			Retry:       retry,
			StatusCode:  0,
			Disposition: Classify(err),
			Err:         err,
		}
		var httpErr *HTTPError
		if errors.As(err, &httpErr) {
			leg.StatusCode = httpErr.StatusCode
		}
		if opts.Hook != nil {
			attempt := identity
			if !start.IsZero() {
				attempt.Duration = time.Since(start)
			}
			attempt.Err = leg
			opts.Hook(attempt)
		}
		return nil, leg
	}

	client := opts.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}

	// The deadline already lives on ctx; this cancel only tears down the
	// connection once the attempt is done.
	reqCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fail(err, time.Time{})
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+chosenKey)
	for name, value := range opts.providerHeaders(prov.Name) {
		req.Header.Set(name, value)
	}

	opts.Logger.Info("sending embedding request",
		"url", url,
		"model", modelName,
		"api_key", previewAPIKey(chosenKey),
		"inputs", len(input),
	)

	start := time.Now().UTC()
	resp, err := client.Do(req)
	if err != nil {
		return fail(err, start)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		return fail(httpError(resp), start)
	}

	var apiResp embeddingAPIResponse
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return fail(fmt.Errorf("decode embedding response: %w", err), start)
	}

	if len(apiResp.Data) != len(input) {
		return fail(fmt.Errorf("embedding response returned %d vectors for %d inputs", len(apiResp.Data), len(input)), start)
	}

	// Sort by index to guarantee input-order alignment.
	sort.Slice(apiResp.Data, func(i, j int) bool {
		return apiResp.Data[i].Index < apiResp.Data[j].Index
	})

	embeddings := make([][]float64, len(apiResp.Data))
	for i, item := range apiResp.Data {
		embeddings[i] = item.Embedding
	}

	total := time.Since(start)
	usage := newUsage(inputTokens, 0, 0, 0, true)
	if apiResp.Usage != nil {
		usage = newUsage(apiResp.Usage.PromptTokens, 0, 0, 0, false)
	}
	opts.Logger.Info(
		formatMetrics(modelName, usage.Input, 0, total, 0),
		"model", modelName,
	)

	if opts.Hook != nil {
		attempt := identity
		attempt.Usage = usage
		attempt.Duration = total
		opts.Hook(attempt)
	}

	return &EmbeddingResponse{
		Embeddings: embeddings,
		Model:      modelSpec,
		ModelName:  modelName,
		Provider:   prov.Name,
		Usage:      usage,
	}, nil
}
