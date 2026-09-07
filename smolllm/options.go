package smolllm

import (
	"log/slog"
	"maps"
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
	// Model accepts provider/model strings; comma-separate multiple entries to try them in order.
	Model string
	// NewSelector builds the model selection strategy for one call. Takes
	// precedence over Model. It is a factory, not a selector: selectors are
	// stateful, so sharing one across calls would let the first call exhaust it.
	NewSelector func() ModelSelector
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
	// LegEfforts overrides ReasoningEffort for individual legs, keyed by the
	// exact model spec the chain names. A leg with no entry uses ReasoningEffort.
	LegEfforts map[string]string
	// Providers supplies credentials per provider name, keyed as the model spec
	// names them. Use BareProvider for bare model specs.
	Providers map[string]ProviderConfig
	// DefaultProvider supplies credentials for any leg without a Providers entry.
	DefaultProvider ProviderConfig
	// ImagePaths embeds local files or data URLs into the first user message.
	ImagePaths []string
	// Timeout bounds the WHOLE call: every retry, every fallback leg, and the
	// consumption of the stream. Zero disables the bound.
	Timeout time.Duration
	// MaxRetries caps the attempts made against one model before the chain
	// advances. 1 disables retrying.
	MaxRetries int
	// RemoveBackticks toggles best-effort markdown fence stripping post-response.
	RemoveBackticks bool
	// HTTPClient allows injecting a custom HTTP client implementation.
	HTTPClient *http.Client
	// Logger captures structured logs. Must not be nil.
	Logger *slog.Logger
	// Hook is called after each LLM call attempt with usage and error details.
	Hook func(Attempt)
	// MinOutputTokens rejects responses shorter than this (estimated tokens).
	// Helps detect context window overflow where models return near-empty output.
	// Only applies when input > 1000 tokens. 0 = disabled (default).
	MinOutputTokens int
	// Dimensions truncates embedding vectors to the given length (Embed only).
	// Requires a model that supports MRL (e.g. text-embedding-3-*, qwen3-embedding).
	// 0 = use the model's native dimensionality (default).
	Dimensions int
	// ExtraBody carries raw request fields the library does not model (e.g. tools,
	// response_format). Merged into the payload last, so the caller wins.
	ExtraBody map[string]any
}

func defaultOptions() Options {
	return Options{
		Model:           "",
		NewSelector:     nil,
		Temperature:     nil,
		TopP:            nil,
		MaxTokens:       nil,
		Stop:            nil,
		Seed:            nil,
		ReasoningEffort: nil,
		LegEfforts:      nil,
		Providers:       nil,
		DefaultProvider: ProviderConfig{BaseURL: "", APIKey: "", Headers: nil},
		ImagePaths:      nil,
		Timeout:         600 * time.Second,
		MaxRetries:      defaultMaxRetries,
		RemoveBackticks: false,
		HTTPClient:      nil,
		Logger:          newDefaultLogger(),
		Hook:            nil,
		MinOutputTokens: 0,
		Dimensions:      0,
		ExtraBody:       nil,
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

// WithModel explicitly selects a provider/model string. Provide comma-separated
// entries (e.g. "gemini/flash,openai/gpt-4o-mini") to list ordered fallbacks.
func WithModel(model string) Option {
	return func(o *Options) {
		o.Model = model
	}
}

// WithModelSet selects randomly from the provided models with equal probability.
// On failure the call advances through the remaining models until exhausted.
func WithModelSet(models ...string) Option {
	if len(models) == 0 {
		panic("WithModelSet: at least one model required")
	}
	// Copy so a later caller mutation cannot reach an in-flight call, and build
	// a fresh selector per call so one call never exhausts the next one's pool.
	copied := make([]string, len(models))
	copy(copied, models)
	return func(o *Options) {
		o.NewSelector = func() ModelSelector { return NewRandomSelector(copied, nil) }
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
	copied := make(map[string]float64, len(weights))
	for m, w := range weights {
		if math.IsNaN(w) || w <= 0 {
			panic("WithModelWeights: weights must be positive and not NaN")
		}
		models = append(models, m)
		copied[m] = w
	}
	return func(o *Options) {
		o.NewSelector = func() ModelSelector { return NewRandomSelector(models, copied) }
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

// WithLegReasoningEffort overrides the reasoning effort for one leg of the
// chain, matched by its exact model spec as written in WithModel or
// WithModelSet (e.g. "groq/openai/gpt-oss-120b"). A leg without an override
// uses WithReasoningEffort. Repeatable; a later call for the same spec wins.
//
// The value still has to clear the leg provider's own allowlist, which Validate
// checks per leg. An override naming a spec the chain does not carry is a
// Validate error rather than a panic: the chain can be set per call, so an
// unmatched override is only knowable once both are in hand.
func WithLegReasoningEffort(model, effort string) Option {
	spec := legSpecKey(model)
	if spec == "" {
		panic("WithLegReasoningEffort: model must not be empty")
	}
	value := strings.ToLower(strings.TrimSpace(effort))
	if value == "" {
		panic("WithLegReasoningEffort: effort must not be empty")
	}
	return func(o *Options) {
		// Copy on write: Options is passed by value, so a per-call option must
		// never reach into the map a Client was built with.
		next := make(map[string]string, len(o.LegEfforts)+1)
		maps.Copy(next, o.LegEfforts)
		next[spec] = value
		o.LegEfforts = next
	}
}

// legSpecKey normalizes a model spec for LegEfforts lookups. The chain trims
// the specs it runs, so an override registered with stray whitespace still has
// to match the leg it names.
func legSpecKey(model string) string {
	return strings.TrimSpace(model)
}

// reasoningEffortFor returns the effort that applies to one leg: its own
// override when WithLegReasoningEffort registered one, otherwise the chain-wide
// WithReasoningEffort, otherwise none. The result is still unvalidated — the
// leg provider's allowlist is applied by normalizeReasoningEffort.
func (o Options) reasoningEffortFor(model string) *string {
	if effort, ok := o.LegEfforts[legSpecKey(model)]; ok {
		return &effort
	}
	return o.ReasoningEffort
}

// BareProvider names the empty provider of a bare model spec (one with no
// "provider/" prefix).
const BareProvider = ""

// ProviderConfig supplies credentials for one provider explicitly, in place of
// environment lookup. BaseURL and APIKey accept comma-separated lists, which the
// balancer rotates across calls.
type ProviderConfig struct {
	BaseURL string
	APIKey  string
	// Headers are applied to the request after Content-Type and Authorization,
	// so a caller header of the same name wins.
	Headers map[string]string
}

// WithProvider supplies credentials for legs of one provider only. Pass
// BareProvider to configure bare model specs. Fields left empty fall through to
// WithDefaultProvider, then the environment, then the provider table; headers
// from both are merged, with this entry winning per key.
func WithProvider(name string, cfg ProviderConfig) Option {
	copied := cfg
	copied.Headers = cloneHeaders(cfg.Headers)
	return func(o *Options) {
		// Copy on write: Options is passed by value, so a per-call option must
		// never reach into the map a Client was built with.
		next := make(map[string]ProviderConfig, len(o.Providers)+1)
		maps.Copy(next, o.Providers)
		next[name] = copied
		o.Providers = next
	}
}

// WithDefaultProvider supplies credentials for any leg with no WithProvider
// entry. Precedence per leg is WithProvider, then WithDefaultProvider, then the
// environment, then the provider table.
func WithDefaultProvider(cfg ProviderConfig) Option {
	copied := cfg
	copied.Headers = cloneHeaders(cfg.Headers)
	return func(o *Options) {
		o.DefaultProvider = copied
	}
}

func cloneHeaders(headers map[string]string) map[string]string {
	if len(headers) == 0 {
		return nil
	}
	return maps.Clone(headers)
}

// providerBaseURL returns the explicitly configured base URL for a provider, or
// the empty string when the caller configured none.
func (o Options) providerBaseURL(name string) string {
	if cfg, ok := o.Providers[name]; ok && strings.TrimSpace(cfg.BaseURL) != "" {
		return cfg.BaseURL
	}
	return o.DefaultProvider.BaseURL
}

// providerAPIKey returns the explicitly configured API key for a provider, or
// the empty string when the caller configured none.
func (o Options) providerAPIKey(name string) string {
	if cfg, ok := o.Providers[name]; ok && strings.TrimSpace(cfg.APIKey) != "" {
		return cfg.APIKey
	}
	return o.DefaultProvider.APIKey
}

// providerHeaders merges the default headers with the provider's own, letting
// the provider entry win per key.
func (o Options) providerHeaders(name string) map[string]string {
	cfg, ok := o.Providers[name]
	if !ok {
		return o.DefaultProvider.Headers
	}
	if len(o.DefaultProvider.Headers) == 0 {
		return cfg.Headers
	}
	merged := maps.Clone(o.DefaultProvider.Headers)
	maps.Copy(merged, cfg.Headers)
	return merged
}

// WithImagePaths attaches user images to the request.
func WithImagePaths(paths ...string) Option {
	copied := make([]string, len(paths))
	copy(copied, paths)
	return func(o *Options) {
		o.ImagePaths = copied
	}
}

// WithTimeout bounds the whole call: every retry, every fallback leg, and the
// consumption of the stream. Zero disables the bound.
func WithTimeout(timeout time.Duration) Option {
	return func(o *Options) {
		o.Timeout = timeout
	}
}

// WithMaxRetries caps the attempts made against one model before the chain
// advances to the next. Pass 1 to disable retrying.
func WithMaxRetries(attempts int) Option {
	if attempts <= 0 {
		panic("WithMaxRetries: attempts must be positive")
	}
	return func(o *Options) {
		o.MaxRetries = attempts
	}
}

// WithBacktickRemoval strips enclosing markdown fences once complete.
func WithBacktickRemoval() Option {
	return func(o *Options) {
		o.RemoveBackticks = true
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

// WithHook registers the per-attempt observation callback. It fires live, as
// each leg finishes, which a long-running server needs.
func WithHook(fn func(Attempt)) Option {
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

// reservedExtraBodyKeys are read back by the library machinery: the stream parser,
// usage collection and routing all depend on them, so a caller override would
// silently break them.
var reservedExtraBodyKeys = []string{"stream", "stream_options", "messages", "model"}

// WithExtraBody sets raw request fields the library does not model, merged into
// the payload last so they win over library defaults. Panics when the caller sets
// a field the library machinery reads back.
func WithExtraBody(fields map[string]any) Option {
	var reserved []string
	for _, key := range reservedExtraBodyKeys {
		if _, ok := fields[key]; ok {
			reserved = append(reserved, key)
		}
	}
	if len(reserved) > 0 {
		panic("WithExtraBody: may not set " + strings.Join(reserved, ", "))
	}
	// Copy so later caller mutations cannot reach an in-flight request.
	copied := make(map[string]any, len(fields))
	for key, value := range fields {
		copied[key] = value
	}
	return func(o *Options) {
		o.ExtraBody = copied
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
