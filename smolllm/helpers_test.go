package smolllm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// testMutated is the sentinel the aliasing tests write into a caller's own slice
// or map after handing it to an Option, to prove the Option copied it.
const testMutated = "mutated"

// testProviderConfig builds a fully populated ProviderConfig so exhaustruct is
// satisfied without every call site repeating the fields it does not set.
func testProviderConfig(baseURL, apiKey string) ProviderConfig {
	return ProviderConfig{BaseURL: baseURL, APIKey: apiKey, Headers: nil}
}

// withTestProvider points every leg of a chain at one base URL and key, which is
// how these offline tests aim all providers at a single httptest server. An
// empty baseURL leaves base-URL resolution to the environment and provider table.
func withTestProvider(baseURL, apiKey string) Option {
	return WithDefaultProvider(testProviderConfig(baseURL, apiKey))
}

// requireAnswered fails the test unless the call produced a usable answer,
// reporting the chain's own explanation when it did not.
func requireAnswered(t *testing.T, msg *AssistantMessage) *AssistantMessage {
	t.Helper()
	require.NotNil(t, msg, "Ask and Result never return nil")
	require.NotEqual(t, StopReasonError, msg.StopReason, "call failed: %s", msg.ErrorMessage)
	require.NotEqual(t, StopReasonAborted, msg.StopReason, "call aborted: %s", msg.ErrorMessage)
	require.True(t, msg.StopReason.Terminal(), "a returned message is always terminal")
	return msg
}

// requireFailed fails the test unless the call ended in an error, which the
// library reports on the message rather than as a Go error.
func requireFailed(t *testing.T, msg *AssistantMessage) *AssistantMessage {
	t.Helper()
	require.NotNil(t, msg, "Ask and Result never return nil")
	require.Equal(t, StopReasonError, msg.StopReason)
	require.NotEmpty(t, msg.ErrorMessage, "a failed call must say why")
	return msg
}

// drain reads a stream to completion and returns its terminal message.
func drain(stream *EventStream) *AssistantMessage {
	for range stream.Events() {
		_ = 0
	}
	return stream.Result()
}

// collect reads a stream to completion, returning every event and the terminal
// message.
func collect(stream *EventStream) ([]Event, *AssistantMessage) {
	var events []Event
	for event := range stream.Events() {
		events = append(events, event)
	}
	return events, stream.Result()
}

// eventsOfKind filters a collected event list.
func eventsOfKind(events []Event, kind EventKind) []Event {
	var matched []Event
	for _, event := range events {
		if event.Kind == kind {
			matched = append(matched, event)
		}
	}
	return matched
}

// deltaText joins the Delta fields of the given events.
func deltaText(events []Event, kind EventKind) string {
	text := ""
	for _, event := range eventsOfKind(events, kind) {
		text += event.Delta
	}
	return text
}
