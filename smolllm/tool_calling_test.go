package smolllm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const weatherArgs = `{"city":"Paris"}`

// writeToolCallStream emulates a provider answering with a tool call and no text:
// the argument JSON arrives fragmented across frames, as every provider streams it.
func writeToolCallStream(t *testing.T, w http.ResponseWriter, finishReason string) {
	t.Helper()

	w.Header().Set("Content-Type", "text/event-stream")
	_, err := fmt.Fprintf(w,
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":"+
			"[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\","+
			"\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"+
			"\"function\":{\"arguments\":\"{\\\"ci\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,"+
			"\"function\":{\"arguments\":\"ty\\\":\\\"Paris\\\"}\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":%q}]}\n\n"+
			"data: [DONE]\n\n",
		finishReason,
	)
	if err != nil {
		t.Errorf("write fake provider response: %v", err)
		return
	}
}

func newToolCallServer(t *testing.T, finishReason string) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeToolCallStream(t, w, finishReason)
	}))
}

// ------------------------------------------------------------------ accumulator

func feedLine(t *testing.T, acc *toolCallAccumulator, payload string) {
	t.Helper()
	_, err := parseChunkLine(newDefaultLogger(), "data: "+payload, nil, nil, acc, nil)
	require.NoError(t, err)
}

func TestToolCallAccumulatorMergesArgumentFragments(t *testing.T) {
	t.Parallel()
	acc := newToolCallAccumulator()
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",`+
		`"function":{"name":"get_weather","arguments":""}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"ci"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 1)
	assert.Equal(t, "call_1", calls[0].ID)
	assert.Equal(t, "function", calls[0].Type)
	assert.Equal(t, "get_weather", calls[0].Function.Name)
	assert.JSONEq(t, weatherArgs, calls[0].Function.Arguments)
}

func TestToolCallAccumulatorKeepsParallelCallsOrdered(t *testing.T) {
	t.Parallel()
	acc := newToolCallAccumulator()
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[`+
		`{"index":1,"id":"b","type":"function","function":{"name":"second"}},`+
		`{"index":0,"id":"a","type":"function","function":{"name":"first"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 2)
	assert.Equal(t, []string{"a", "b"}, []string{calls[0].ID, calls[1].ID})
	assert.Equal(t, []int{0, 1}, acc.sortedIndexes())
	assert.Equal(t, "first", calls[0].Function.Name)
	assert.Equal(t, "{}", calls[0].Function.Arguments)
	assert.Empty(t, calls[1].Function.Arguments)
}

func TestToolCallAccumulatorWithoutIndexStartsNewCallOnID(t *testing.T) {
	t.Parallel()
	acc := newToolCallAccumulator()
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"id":"a","type":"function","function":{"name":"f"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"function":{"arguments":"{\"x\":1}"}}]}}]}`)
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"id":"b","type":"function","function":{"name":"g"}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 2)
	assert.Equal(t, []string{"a", "b"}, []string{calls[0].ID, calls[1].ID})
	assert.Equal(t, `{"x":1}`, calls[0].Function.Arguments)
}

func TestToolCallAccumulatorIsEmptyForPlainTextStream(t *testing.T) {
	t.Parallel()
	acc := newToolCallAccumulator()
	feedLine(t, acc, `{"choices":[{"delta":{"content":"hi"}}]}`)
	assert.Empty(t, acc.result())
	assert.Empty(t, acc.sortedIndexes())
}

func TestToolCallAccumulatorKeepsProviderExtras(t *testing.T) {
	t.Parallel()
	acc := newToolCallAccumulator()
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",`+
		`"function":{"name":"get_weather","arguments":"{}"},`+
		`"extra_content":{"google":{"thought_signature":"sig-abc"}}}]}}]}`)

	calls := acc.result()
	require.Len(t, calls, 1)
	require.Contains(t, calls[0].Extra, "extra_content")
	assert.JSONEq(t, `{"google":{"thought_signature":"sig-abc"}}`, string(calls[0].Extra["extra_content"]))
}

// A snapshot taken mid-stream shows the arguments received so far, and must not
// change when later fragments arrive.
func TestToolCallSnapshotIsIndependentOfLaterFragments(t *testing.T) {
	t.Parallel()
	acc := newToolCallAccumulator()
	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function",`+
		`"function":{"name":"get_weather","arguments":"{\"ci"}}]}}]}`)

	partial := acc.snapshot()
	require.Len(t, partial, 1)
	assert.Equal(t, `{"ci`, partial[0].Function.Arguments)

	feedLine(t, acc, `{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]}}]}`)

	assert.Equal(t, `{"ci`, partial[0].Function.Arguments, "an earlier snapshot never changes")
	assert.JSONEq(t, weatherArgs, acc.snapshot()[0].Function.Arguments)
}

func TestToolCallRoundTripsExtrasThroughJSON(t *testing.T) {
	t.Parallel()
	raw := `{"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"},` +
		`"extra_content":{"google":{"thought_signature":"sig"}}}`

	var call ToolCall
	require.NoError(t, json.Unmarshal([]byte(raw), &call))
	assert.Equal(t, "call_1", call.ID)

	encoded, err := json.Marshal(call)
	require.NoError(t, err)
	assert.JSONEq(t, raw, string(encoded))
}

// ------------------------------------------------------------------ request side

func TestRequestValidateAcceptsToolMessages(t *testing.T) {
	t.Parallel()
	req := RequestFromMessages([]Message{
		User("weather in Paris?"),
		AssistantToolCalls("", []ToolCall{{
			ID: "call_1", Type: "function",
			Function: ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
			Extra:    nil,
		}}),
		ToolResult("call_1", `{"temp_c":18}`),
	})
	require.NoError(t, req.Validate())
}

func TestRequestValidateStillRejectsFunctionRole(t *testing.T) {
	t.Parallel()
	//nolint:exhaustruct,staticcheck // the deprecated function role is exactly what this test pins as rejected
	legacy := openai.ChatCompletionFunctionMessageParam{
		Content: openai.String("x"),
		Name:    "f",
	}
	msg := Message{OfFunction: &legacy} //nolint:exhaustruct // union arm under test
	err := Request{System: "", Tools: nil, Messages: []Message{msg}}.Validate()
	require.ErrorContains(t, err, "unsupported role")
}

// Typed tools reach the wire in the OpenAI-compatible shape, with the JSON
// Schema passed through untouched.
func TestRequestToolsReachTheWire(t *testing.T) {
	t.Parallel()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		writeChatSuccess(t, w, "sunny", "stop")
	}))
	defer srv.Close()

	schema := json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`)
	req := RequestFromString("weather in Paris?")
	req.Tools = []Tool{{
		Name:        "get_weather",
		Description: "Look up the current weather for a city",
		Parameters:  schema,
	}}

	msg := Ask(context.Background(), req,
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireAnswered(t, msg)

	tools, ok := body["tools"].([]any)
	require.True(t, ok, "tools must reach the wire")
	require.Len(t, tools, 1)
	tool, ok := tools[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "function", tool["type"])
	function, ok := tool["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "get_weather", function["name"])
	assert.Equal(t, "Look up the current weather for a city", function["description"])

	encoded, err := json.Marshal(function["parameters"])
	require.NoError(t, err)
	assert.JSONEq(t, string(schema), string(encoded), "the schema is passed through untouched")
}

func TestRequestToolsAreOmittedWhenEmpty(t *testing.T) {
	t.Parallel()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		writeChatSuccess(t, w, "hi", "stop")
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireAnswered(t, msg)
	assert.NotContains(t, body, "tools")
}

// The escape hatch still merges last, so a caller who sets tools there wins over
// the typed field.
func TestExtraBodyToolsWinOverTypedTools(t *testing.T) {
	t.Parallel()

	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		writeChatSuccess(t, w, "hi", "stop")
	}))
	defer srv.Close()

	req := RequestFromString("hi")
	req.Tools = []Tool{{Name: "typed", Description: "", Parameters: nil}}

	msg := Ask(context.Background(), req,
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"),
		WithExtraBody(map[string]any{"tools": []any{
			map[string]any{"type": "function", "function": map[string]any{"name": "from_extra_body"}},
		}}))
	requireAnswered(t, msg)

	tools, ok := body["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1)
	tool, ok := tools[0].(map[string]any)
	require.True(t, ok)
	function, ok := tool["function"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "from_extra_body", function["name"])
}

func TestReplayedToolConversationReachesTheWire(t *testing.T) {
	t.Parallel()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		writeChatSuccess(t, w, "It is 18C in Paris.", "stop")
	}))
	defer srv.Close()

	signature := json.RawMessage(`{"google":{"thought_signature":"sig"}}`)
	req := RequestFromMessages([]Message{
		User("weather in Paris?"),
		AssistantToolCalls("", []ToolCall{{
			ID: "call_1", Type: "function",
			Function: ToolCallFunction{Name: "get_weather", Arguments: weatherArgs},
			Extra:    map[string]json.RawMessage{"extra_content": signature},
		}}),
		ToolResult("call_1", `{"temp_c":18}`),
	})

	msg := Ask(context.Background(), req,
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireAnswered(t, msg)
	assert.Equal(t, "It is 18C in Paris.", msg.Content)

	messages, ok := body["messages"].([]any)
	require.True(t, ok)
	require.Len(t, messages, 3)

	assistant, ok := messages[1].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "assistant", assistant["role"])
	calls, ok := assistant["tool_calls"].([]any)
	require.True(t, ok)
	require.Len(t, calls, 1)
	call, ok := calls[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "call_1", call["id"])
	assert.Equal(t, map[string]any{"google": map[string]any{"thought_signature": "sig"}}, call["extra_content"])

	toolMsg, ok := messages[2].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "call_1", toolMsg["tool_call_id"])
	assert.Equal(t, `{"temp_c":18}`, toolMsg["content"])
}

// ------------------------------------------------------------------ end to end

func TestAskReturnsToolCallsWithoutContent(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "tool_calls")
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireAnswered(t, msg)

	assert.Empty(t, msg.Content)
	assert.Equal(t, "tool_calls", msg.FinishReason)
	assert.Equal(t, StopReasonToolUse, msg.StopReason)
	require.Len(t, msg.ToolCalls, 1)
	assert.Equal(t, "get_weather", msg.ToolCalls[0].Function.Name)
	assert.JSONEq(t, weatherArgs, msg.ToolCalls[0].Function.Arguments)
}

// Tool calls alone mean tool use even when the provider labelled the turn
// "stop", which is what Gemini does.
func TestToolCallsImplyToolUseStopReason(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "stop")
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("weather?"),
		WithModel("gemini/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireAnswered(t, msg)

	assert.Equal(t, "stop", msg.FinishReason, "the provider string is kept verbatim")
	assert.Equal(t, StopReasonToolUse, msg.StopReason)
	require.Len(t, msg.ToolCalls, 1)
}

// This inverts the v0.2 invariant: argument fragments now reach consumers as
// they stream, and the complete call arrives on tool_call_end.
func TestStreamPushesToolCallFragments(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "tool_calls")
	defer srv.Close()

	events, msg := collect(Stream(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k")))
	requireAnswered(t, msg)

	starts := eventsOfKind(events, EventToolCallStart)
	require.Len(t, starts, 1, "one slot was opened")
	assert.Equal(t, 0, starts[0].Index)

	fragments := eventsOfKind(events, EventToolCallDelta)
	require.NotEmpty(t, fragments, "partial tool-call fragments now reach consumers")
	for _, fragment := range fragments {
		assert.Equal(t, 0, fragment.Index)
	}
	assert.JSONEq(t, weatherArgs, deltaText(events, EventToolCallDelta))

	ends := eventsOfKind(events, EventToolCallEnd)
	require.Len(t, ends, 1, "the complete call arrives once")
	require.NotNil(t, ends[0].ToolCall)
	assert.Equal(t, "get_weather", ends[0].ToolCall.Function.Name)
	assert.JSONEq(t, weatherArgs, ends[0].ToolCall.Function.Arguments)
	assert.Equal(t, 0, ends[0].Index)

	require.Len(t, msg.ToolCalls, 1)
	assert.JSONEq(t, weatherArgs, msg.ToolCalls[0].Function.Arguments)
}

// A leg the guards reject must not announce completed calls.
func TestTruncatedToolCallsNeverReachToolCallEnd(t *testing.T) {
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

func TestAskFailsLegWhenToolCallsAreTruncated(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "length")
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireFailed(t, msg)
	assert.Contains(t, msg.ErrorMessage, "truncated")
}

func TestAskSkipsMinOutputTokensForToolCalls(t *testing.T) {
	t.Parallel()
	srv := newToolCallServer(t, "tool_calls")
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("weather?"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"),
		WithMinOutputTokens(500))
	requireAnswered(t, msg)
	require.Len(t, msg.ToolCalls, 1)
}

func TestAskStillFailsOnEmptyResponseWithoutToolCalls(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, err := fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+"data: [DONE]\n\n")
		if err != nil {
			t.Errorf("write fake provider response: %v", err)
		}
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireFailed(t, msg)
	assert.Contains(t, msg.ErrorMessage, "empty response")
}
