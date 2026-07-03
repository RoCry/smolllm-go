package smolllm

import (
	"context"
	"log/slog"
	"math"
	"net/http"
	"os"
	"strings"
	"time"
)

// Option mutates the configuration for Ask or Stream.
type Option func(*Options)

// Options bundles optional arguments for Ask and Stream.
type Options struct {
	// SystemPrompt becomes the leading system message when populated.
	SystemPrompt string
	// Model accepts provider/model strings; comma-separate multiple entries to try them in order.
	Model string
	// Selector provides custom model selection strategy. Takes precedence over Model.
	Selector ModelSelector
	// Temperature controls sampling randomness.
	Temperature *float64
	// TopP applies nucleus sampling cutoff.
	TopP *float64
	// MaxTokens limits output length for providers that support it.
	MaxTokens *int
	// Stop sequences terminate generation for providers that support them.
	Stop []string
	// Seed asks providers that support deterministic sampling to use this seed.
	Seed *int
	// ReasoningEffort controls how much thinking a reasoning model does. Passed through to the provider as-is.
	ReasoningEffort *string
	// APIKey overrides env lookup. Comma-separated values enable automatic rotation across calls.
	APIKey string
	// BaseURL overrides the inferred endpoint for the provider.
	BaseURL string
	// ImagePaths embeds local files or data URLs into the first user message.
	ImagePaths []string
	// Timeout bounds the total duration including retries.
	Timeout time.Duration
	// RemoveBackticks toggles best-effort markdown fence stripping post-response.
	RemoveBackticks bool
	// StreamHandler receives streamed deltas when streaming is enabled.
	StreamHandler func(context.Context, string) error
	// HTTPClient allows injecting a custom HTTP client implementation.
	HTTPClient *http.Client
	// Logger captures structured logs. Must not be nil.
	Logger *slog.Logger
	// Hook is called after each LLM call attempt with usage and error details.
	Hook func(RequestEvent)
	// MinOutputTokens rejects responses shorter than this (estimated tokens).
	// Helps detect context window overflow where models return near-empty output.
	// Only applies when input > 1000 tokens. 0 = disabled (default).
	MinOutputTokens int
	// Dimensions truncates embedding vectors to the given length (Embed only).
	// Requires a model that supports MRL (e.g. text-embedding-3-*, qwen3-embedding).
	// 0 = use the model's native dimensionality (default).
	Dimensions int
}

func defaultOptions() Options {
	return Options{
		SystemPrompt:    "",
		Model:           "",
		Selector:        nil,
		Temperature:     nil,
		TopP:            nil,
		MaxTokens:       nil,
		Stop:            nil,
		Seed:            nil,
		ReasoningEffort: nil,
		APIKey:          "",
		BaseURL:         "",
		ImagePaths:      nil,
		Timeout:         600 * time.Second,
		RemoveBackticks: false,
		StreamHandler:   nil,
		HTTPClient:      nil,
		Logger:          newDefaultLogger(),
		Hook:            nil,
		MinOutputTokens: 0,
		Dimensions:      0,
	}
}

// applyOptions creates default options and applies all provided option functions.
func applyOptions(opts ...Option) Options {
	options := defaultOptions()
	for _, opt := range opts {
		opt(&options)
	}
	return options
}

// WithSystemPrompt sets the system prompt.
func WithSystemPrompt(system string) Option {
	return func(o *Options) {
		o.SystemPrompt = system
	}
}

// WithModel explicitly selects a provider/model string. Provide comma-separated
// entries (e.g. "gemini/flash,openai/gpt-4o-mini") to list ordered fallbacks.
func WithModel(model string) Option {
	return func(o *Options) {
		o.Model = model
	}
}

// WithModelSet selects randomly from the provided models with equal probability.
// On failure, retries remaining models until exhausted.
func WithModelSet(models ...string) Option {
	if len(models) == 0 {
		panic("WithModelSet: at least one model required")
	}
	return func(o *Options) {
		o.Selector = NewRandomSelector(models, nil)
	}
}

// WithModelWeights selects randomly using the provided weights.
// Higher weights increase selection probability. Weights must be positive.
// On failure, retries remaining models with re-normalized weights.
func WithModelWeights(weights map[string]float64) Option {
	if len(weights) == 0 {
		panic("WithModelWeights: at least one model required")
	}
	models := make([]string, 0, len(weights))
	for m, w := range weights {
		if math.IsNaN(w) || w <= 0 {
			panic("WithModelWeights: weights must be positive and not NaN")
		}
		models = append(models, m)
	}
	return func(o *Options) {
		o.Selector = NewRandomSelector(models, weights)
	}
}

// WithTemperature sets the sampling temperature. Valid range is [0, 2].
func WithTemperature(value float64) Option {
	if math.IsNaN(value) || value < 0 || value > 2 {
		panic("WithTemperature: value must be between 0 and 2 inclusive")
	}
	return func(o *Options) {
		v := value
		o.Temperature = &v
	}
}

// WithTopP sets the nucleus sampling probability mass. Valid range is [0, 1].
func WithTopP(value float64) Option {
	if math.IsNaN(value) || value < 0 || value > 1 {
		panic("WithTopP: value must be between 0 and 1 inclusive")
	}
	return func(o *Options) {
		v := value
		o.TopP = &v
	}
}

// WithMaxTokens sets the maximum number of output tokens.
func WithMaxTokens(value int) Option {
	if value <= 0 {
		panic("WithMaxTokens: value must be positive")
	}
	return func(o *Options) {
		v := value
		o.MaxTokens = &v
	}
}

// WithStop sets one or more stop sequences.
func WithStop(stops ...string) Option {
	if len(stops) == 0 {
		panic("WithStop: at least one stop sequence required")
	}
	copied := make([]string, len(stops))
	for i, stop := range stops {
		trimmed := strings.TrimSpace(stop)
		if trimmed == "" {
			panic("WithStop: stop sequences must not be empty")
		}
		copied[i] = stop
	}
	return func(o *Options) {
		o.Stop = copied
	}
}

// WithSeed sets a deterministic sampling seed for providers that support it.
func WithSeed(value int) Option {
	return func(o *Options) {
		v := value
		o.Seed = &v
	}
}

// WithReasoningEffort sets the reasoning effort for reasoning models. Value is passed through to the provider as-is.
func WithReasoningEffort(value string) Option {
	v := strings.ToLower(strings.TrimSpace(value))
	if v == "" {
		panic("WithReasoningEffort: value must not be empty")
	}
	return func(o *Options) {
		o.ReasoningEffort = &v
	}
}

// WithAPIKey overrides the resolved API key. Multiple comma-separated keys are
// balanced automatically just like the environment variable format.
func WithAPIKey(key string) Option {
	return func(o *Options) {
		o.APIKey = key
	}
}

// WithBaseURL overrides the resolved base URL.
func WithBaseURL(url string) Option {
	return func(o *Options) {
		o.BaseURL = url
	}
}

// WithImagePaths attaches user images to the request.
func WithImagePaths(paths ...string) Option {
	copied := make([]string, len(paths))
	copy(copied, paths)
	return func(o *Options) {
		o.ImagePaths = copied
	}
}

// WithTimeout defines the hard timeout for the request.
func WithTimeout(timeout time.Duration) Option {
	return func(o *Options) {
		o.Timeout = timeout
	}
}

// WithBacktickRemoval strips enclosing markdown fences once complete.
func WithBacktickRemoval() Option {
	return func(o *Options) {
		o.RemoveBackticks = true
	}
}

// WithStreamHandler registers a callback for streamed deltas.
func WithStreamHandler(handler func(context.Context, string) error) Option {
	return func(o *Options) {
		o.StreamHandler = handler
	}
}

// WithHTTPClient injects a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(o *Options) {
		o.HTTPClient = client
	}
}

// WithLogger overrides the logger. Logger must not be nil.
func WithLogger(logger *slog.Logger) Option {
	if logger == nil {
		panic("WithLogger: logger must not be nil")
	}
	return func(o *Options) {
		o.Logger = logger
	}
}

// WithHook registers a callback invoked after each LLM call attempt.
func WithHook(fn func(RequestEvent)) Option {
	return func(o *Options) {
		o.Hook = fn
	}
}

// WithMinOutputTokens rejects responses shorter than minTokens (estimated).
// Useful for detecting context window overflow where a model returns near-empty output.
// Only enforced when input exceeds 1000 tokens to allow short replies on small prompts.
func WithMinOutputTokens(minTokens int) Option {
	return func(o *Options) {
		o.MinOutputTokens = minTokens
	}
}

// WithDimensions truncates embedding vectors to the given length. Only applies
// to Embed calls and requires a model that supports MRL (Matryoshka Representation
// Learning), e.g. OpenAI text-embedding-3-* or qwen3-embedding.
func WithDimensions(dimensions int) Option {
	if dimensions <= 0 {
		panic("WithDimensions: dimensions must be positive")
	}
	return func(o *Options) {
		o.Dimensions = dimensions
	}
}

func newDefaultLogger() *slog.Logger {
	level := slog.LevelInfo
	switch strings.ToUpper(os.Getenv("LOG_LEVEL")) {
	case "DEBUG":
		level = slog.LevelDebug
	case "WARN", "WARNING":
		level = slog.LevelWarn
	case "ERROR":
		level = slog.LevelError
	}

	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		AddSource: false,
		Level:     level,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.TimeKey {
				return slog.Attr{
					Key:   attr.Key,
					Value: slog.StringValue(attr.Value.Time().UTC().Format(time.RFC3339)),
				}
			}
			return attr
		},
	})

	return slog.New(handler)
}
