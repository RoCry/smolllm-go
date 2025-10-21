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
	SystemPrompt    string
	Model           string
	APIKey          string
	BaseURL         string
	ImagePaths      []string
	Timeout         time.Duration
	RemoveBackticks bool
	StreamHandler   func(context.Context, string) error
	HTTPClient      *http.Client
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

// WithModel explicitly selects a provider/model string.
func WithModel(model string) Option {
	return func(o *Options) {
		o.Model = model
	}
}

// WithAPIKey overrides the resolved API key.
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
