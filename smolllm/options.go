package smolllm

import (
	"context"
	"net/http"
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
}

func defaultOptions() Options {
	return Options{
		Timeout: 120 * time.Second,
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
