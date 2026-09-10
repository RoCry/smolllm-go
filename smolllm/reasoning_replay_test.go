package smolllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// DeepSeek thinking mode rejects a replayed tool-call turn it did not issue
// unless reasoning_content is present, so attached empty reasoning must still
// reach the wire.
func TestAttachReasoningReachesTheWire(t *testing.T) {
	t.Parallel()
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		writeChatSuccess(t, w, "You are welcome.", "stop")
	}))
	defer srv.Close()

	call := ToolCall{
		ID: testCallID, Type: toolTypeFunction,
		Function: ToolCallFunction{Name: testToolName, Arguments: weatherArgs}, Extra: nil,
	}
	req := RequestFromMessages([]Message{
		User("weather in Paris?"),
		AttachReasoning(AssistantToolCalls("", []ToolCall{call}), ""),
		ToolResult(testCallID, `{"temp_c":18}`),
		AttachReasoning(Assistant("It is 18C in Paris."), "the tool said 18C"),
		User("thanks"),
	})

	msg := Ask(context.Background(), req,
		WithModel("openai/model-a"), withTestProvider(srv.URL+"/", "k"))
	requireAnswered(t, msg)

	messages, ok := body["messages"].([]any)
	require.True(t, ok)
	require.Len(t, messages, 5)

	toolTurn, ok := messages[1].(map[string]any)
	require.True(t, ok)
	reasoning, present := toolTurn["reasoning_content"]
	require.True(t, present, "empty reasoning is still sent")
	assert.Empty(t, reasoning)
	assert.NotEmpty(t, toolTurn["tool_calls"], "attaching keeps the calls")

	answer, ok := messages[3].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "the tool said 18C", answer["reasoning_content"])
	assert.Equal(t, "It is 18C in Paris.", answer["content"])

	user, ok := messages[0].(map[string]any)
	require.True(t, ok)
	assert.NotContains(t, user, "reasoning_content")
}

func TestAttachReasoningPanicsOnNonAssistantMessage(t *testing.T) {
	t.Parallel()
	require.PanicsWithValue(t, "AttachReasoning: message is not an assistant message", func() {
		AttachReasoning(User("hi"), "thinking")
	})
}
