package smolllm

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func extraBodyOptions(extra map[string]any) chatPayloadOptions {
	opts := chatOptions(true)
	opts.ExtraBody = extra
	return opts
}

func TestExtraBodyFieldsReachThePayload(t *testing.T) {
	t.Parallel()
	tools := []any{map[string]any{"type": "function", "function": map[string]any{"name": "get_weather"}}}
	_, body, _, err := buildRequestPayload(
		PromptFromString("hi"), "", "m", "openai", "https://api.openai.com", nil,
		extraBodyOptions(map[string]any{"tools": tools, "tool_choice": "auto", "max_tokens": 256}),
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.Equal(t, "auto", payload["tool_choice"])
	assert.Equal(t, float64(256), payload["max_tokens"])
	require.Len(t, payload["tools"], 1)
}

func TestExtraBodyWinsOverLibraryDefaults(t *testing.T) {
	t.Parallel()
	opts := extraBodyOptions(map[string]any{"temperature": 1.5})
	temperature := 0.1
	opts.Temperature = &temperature

	_, body, _, err := buildRequestPayload(
		PromptFromString("hi"), "", "m", "openai", "https://api.openai.com", nil, opts,
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	assert.InEpsilon(t, 1.5, payload["temperature"], 1e-9)
}

func TestExtraBodyCountsTowardEstimatedTokens(t *testing.T) {
	t.Parallel()
	plain := chatOptions(true)
	_, _, plainTokens, err := buildRequestPayload(
		PromptFromString("hi"), "", "m", "openai", "https://api.openai.com", nil, plain,
	)
	require.NoError(t, err)

	withTools := extraBodyOptions(map[string]any{
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "get_weather", "description": "Look up the current weather for a city",
		}}},
	})
	_, _, toolTokens, err := buildRequestPayload(
		PromptFromString("hi"), "", "m", "openai", "https://api.openai.com", nil, withTools,
	)
	require.NoError(t, err)
	assert.Greater(t, toolTokens, plainTokens)
}

func TestExtraBodyIsAbsentWhenNotSet(t *testing.T) {
	t.Parallel()
	_, body, _, err := buildRequestPayload(
		PromptFromString("hi"), "", "m", "openai", "https://api.openai.com", nil, chatOptions(true),
	)
	require.NoError(t, err)

	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	_, ok := payload["tools"]
	assert.False(t, ok)
}

func TestWithExtraBodyRejectsReservedKeys(t *testing.T) {
	t.Parallel()
	for _, key := range []string{"stream", "stream_options", "messages", "model"} {
		assert.PanicsWithValue(
			t,
			"WithExtraBody: may not set "+key,
			func() { WithExtraBody(map[string]any{key: "x"}) },
		)
	}
}

func TestWithExtraBodyNamesEveryReservedOffender(t *testing.T) {
	t.Parallel()
	assert.PanicsWithValue(
		t,
		"WithExtraBody: may not set messages, model",
		func() { WithExtraBody(map[string]any{"model": "x", "messages": "y", "tools": "z"}) },
	)
}

func TestWithExtraBodyDoesNotAliasCallerMap(t *testing.T) {
	t.Parallel()
	caller := map[string]any{"tools": "original"}
	options := applyOptions(WithExtraBody(caller))
	caller["tools"] = "mutated"
	assert.Equal(t, "original", options.ExtraBody["tools"])
}
