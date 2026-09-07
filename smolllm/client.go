package smolllm

import "sync"

// Client issues chat and embedding calls. It is safe for concurrent use. Each
// Client owns its balancer, so clients built for different model roles never
// share key rotation state.
type Client struct {
	opts     Options
	balancer *simpleBalancer
}

// New builds a Client from the given options.
func New(opts ...Option) *Client {
	return &Client{
		opts:     applyOptions(opts...),
		balancer: newBalancer(),
	}
}

// callOptions layers per-call options over the Client's configured options, so
// a per-call option always wins.
func (c *Client) callOptions(opts ...Option) Options {
	merged := c.opts
	for _, opt := range opts {
		opt(&merged)
	}
	return merged
}

// sharedClient backs the package-level entry points. It is built once so that
// balancer rotation state survives across calls, matching the process-global
// balancer this package used before Client existed.
var sharedClient = sync.OnceValue(func() *Client { return New() })
