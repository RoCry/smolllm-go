package smolllm

import (
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

// Fixtures repeated across these tests. Named rather than spelled out at every
// site, so the value says what it stands for and one edit moves all of them.
const (
	// testChatModel is the chat leg the fake-provider tests aim at.
	testChatModel = "openai/gpt-5"
	// testToolName is the function the tool-calling tests declare and expect back.
	testToolName = "get_weather"
	// testCallID is the id the tool-calling tests give a replayed call.
	testCallID = "call_1"
	// testImageDataURL is a throwaway data URL for the multimodal path.
	testImageDataURL = "data:image/png;base64,AA=="
	// testUnavailableBody is the body a fake HTTP 503 answers with.
	testUnavailableBody = "unavailable"
)

// Stop sequences the option and payload round-trips carry.
const (
	testStopEnd  = "END"
	testStopStop = "STOP"
)

// Wire field names the tool tests assemble request bodies from, and read back
// out of decoded ones, by hand. The tool type itself is toolTypeFunction.
const (
	wireFieldTools    = "tools"
	wireFieldType     = "type"
	wireFieldName     = "name"
	wireFieldFunction = "function"
)

// testMutated is the sentinel the aliasing tests write into a caller's own slice
// or map after handing it to an Option, to prove the Option copied it.
const testMutated = "mutated"

// writeFakeResponse emits a fake provider's response body, one part per write
// so a caller can keep its SSE frames separate. A write that does not land
// means the rig is broken, so it fails the test: t.Errorf and not require,
// because handlers run off the test goroutine where FailNow is not allowed.
func writeFakeResponse(t *testing.T, w io.Writer, parts ...string) {
	t.Helper()

	for _, part := range parts {
		if _, err := io.WriteString(w, part); err != nil {
			t.Errorf("write fake provider response: %v", err)
			return
		}
	}
}

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
