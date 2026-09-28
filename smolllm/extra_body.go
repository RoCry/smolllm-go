package smolllm

import (
	"maps"
	"strings"
)

// reservedExtraBodyKeys are read back by the library machinery: the stream parser,
// usage collection and routing all depend on them, so a caller override would
// silently break them.
var reservedExtraBodyKeys = []string{"stream", "stream_options", "messages", "model"}

// WithExtraBody sets raw request fields the library does not model, merged into
// the payload last so they win over library defaults. Panics when the caller sets
// a field the library machinery reads back.
func WithExtraBody(fields map[string]any) Option {
	copied := checkedExtraBody("WithExtraBody", fields)
	return func(o *Options) {
		o.ExtraBody = copied
	}
}

// checkedExtraBody panics on a reserved key and returns a copy, so later caller
// mutations cannot reach an in-flight request.
func checkedExtraBody(option string, fields map[string]any) map[string]any {
	var reserved []string
	for _, key := range reservedExtraBodyKeys {
		if _, ok := fields[key]; ok {
			reserved = append(reserved, key)
		}
	}
	if len(reserved) > 0 {
		panic(option + ": may not set " + strings.Join(reserved, ", "))
	}
	return maps.Clone(fields)
}

// WithProviderExtraBody sets raw request fields sent only to legs of one
// provider, merged over WithExtraBody. Use it for fields only some providers
// accept, so a fallback leg elsewhere never sees them. Repeatable per provider;
// a later call for the same provider replaces its fields. Reserved keys panic
// as in WithExtraBody.
func WithProviderExtraBody(provider string, fields map[string]any) Option {
	name := strings.TrimSpace(provider)
	if name == "" {
		panic("WithProviderExtraBody: provider must not be empty")
	}
	copied := checkedExtraBody("WithProviderExtraBody", fields)
	return func(o *Options) {
		// Copy on write, as for LegEfforts: never mutate a Client's own map.
		next := make(map[string]map[string]any, len(o.ProviderExtraBody)+1)
		maps.Copy(next, o.ProviderExtraBody)
		next[name] = copied
		o.ProviderExtraBody = next
	}
}

// extraBodyFor returns the extra fields for one provider's legs: the chain-wide
// ones overlaid with that provider's own.
func (o Options) extraBodyFor(provider string) map[string]any {
	own, ok := o.ProviderExtraBody[provider]
	if !ok {
		return o.ExtraBody
	}
	merged := make(map[string]any, len(o.ExtraBody)+len(own))
	maps.Copy(merged, o.ExtraBody)
	maps.Copy(merged, own)
	return merged
}
