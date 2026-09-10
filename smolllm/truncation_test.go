package smolllm

import (
	"bytes"
	"context"
	"log/slog"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Without a declared max_tokens the cut came from a provider or relay default,
// so the leg fails and announces no completed calls.
func TestTruncatedToolCallsWithoutDeclaredMaxTokensNeverReachToolCallEnd(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "length")
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k")))
	requireFailed(t, msg)

	assert.Contains(t, msg.ErrorMessage, "truncated")
	assert.Empty(t, eventsOfKind(events, EventToolCallEnd))
	assert.NotEmpty(t, eventsOfKind(events, EventToolCallDelta), "fragments still streamed before the guard ran")
}

func TestAskFailsLegWhenToolCallsAreTruncatedWithoutDeclaredMaxTokens(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "length")
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireFailed(t, msg)
	assert.Contains(t, msg.ErrorMessage, "truncated")
}

// A max_tokens the caller declared cuts every leg at the same place, so the
// chain hands the calls back instead of advancing to a leg that would fail
// identically.
func TestAskReturnsToolCallsTruncatedAtDeclaredMaxTokens(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "length")
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a,openai/model-b"), withTestProvider(srv.URL+"/", "k"),
		WithMaxTokens(64))

	require.Equal(t, StopReasonLength, msg.StopReason, "length wins over tool calls: %s", msg.ErrorMessage)
	assert.Empty(t, msg.ErrorMessage)
	require.Len(t, msg.ToolCalls, 1)
	assert.Equal(t, testToolName, msg.ToolCalls[0].Function.Name)
	assert.JSONEq(t, weatherArgs, msg.ToolCalls[0].Function.Arguments, "arguments as streamed")
	assert.Len(t, msg.Attempts, 1, "the second leg must not be tried")
}

// An operator reading the log has to see which Tool outgrew the output budget
// and by how much, whichever way the truncation was handled.
func TestTruncatedToolCallsAreNamedWithTheirArgumentBytes(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "length")
	t.Cleanup(srv.Close)
	const named = "get_weather 16 bytes"

	t.Run("in the leg error without a declared max_tokens", func(t *testing.T) {
		t.Parallel()
		msg := Ask(context.Background(), RequestFromString("weather?"),
			WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
		requireFailed(t, msg)
		assert.Contains(t, msg.ErrorMessage, named)
	})

	t.Run("in a warning at a declared max_tokens", func(t *testing.T) {
		t.Parallel()
		var logs lockedBuffer
		logger := slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{
			AddSource: false, Level: slog.LevelWarn, ReplaceAttr: nil,
		}))
		msg := Ask(context.Background(), RequestFromString("weather?"),
			WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"),
			WithMaxTokens(64), WithLogger(logger))
		require.Equal(t, StopReasonLength, msg.StopReason, msg.ErrorMessage)
		assert.Contains(t, logs.String(), named)
		assert.Contains(t, logs.String(), "max_tokens=64")
	})
}

// lockedBuffer collects log output the chain writes from its own goroutine.
type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

func TestStreamAnnouncesToolCallsTruncatedAtDeclaredMaxTokens(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "length")
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"), WithMaxTokens(64)))

	require.Equal(t, StopReasonLength, msg.StopReason, msg.ErrorMessage)
	ends := eventsOfKind(events, EventToolCallEnd)
	require.Len(t, ends, 1, "a streaming consumer sees the same calls Ask returns")
	assert.Equal(t, testToolName, ends[0].ToolCall.Function.Name)
	assert.Len(t, eventsOfKind(events, EventDone), 1)
}
