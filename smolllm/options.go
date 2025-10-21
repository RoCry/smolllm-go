package smolllm

import (
	"context"
	"log/slog"
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
}

func defaultOptions() Options {
	return Options{
		Timeout: 120 * time.Second,
		Logger:  newDefaultLogger(),
	}
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
		Level: level,
		ReplaceAttr: func(groups []string, attr slog.Attr) slog.Attr {
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
