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

// smolayer streams each Codex Reasoning Item as its own delta, ahead of the
// answer, as `delta.reasoning_items: [token]` (src/openai/from_turn.rs). The
// tokens are opaque and must come out byte for byte, in arrival order.
func TestReasoningItemsSurviveDecodeAndEncode(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeFakeResponse(t, w,
			`data: {"choices":[{"index":0,"delta":{"reasoning_items":["rsn_AAA.b64+/="]}}]}`+"\n\n",
			`data: {"choices":[{"index":0,"delta":{"reasoning_items":["rsn_BBB"]}}]}`+"\n\n",
			`data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}`+"\n\n",
			`data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`+"\n\n",
			"data: [DONE]\n\n",
		)
	}))
	defer srv.Close()

	msg := requireAnswered(t, Ask(context.Background(), RequestFromMessages([]Message{User("hi")}),
		WithModel(testChatModel), withTestProvider(srv.URL+"/", "k")))
	assert.Equal(t, "ok", msg.Content)
	assert.Equal(t, []string{"rsn_AAA.b64+/=", "rsn_BBB"}, msg.ReasoningItems)

	encoded, err := json.Marshal(msg)
	require.NoError(t, err)
	assert.Contains(t, string(encoded), `"reasoning_items":["rsn_AAA.b64+/=","rsn_BBB"]`)
}

// A leg that streamed a Reasoning Item and then failed before answering hands
// over to the next leg. Its token belongs to a provider that did not answer, so
// echoing it back would be rejected: only the winner's tokens may survive.
func TestReasoningItemsFromAFailedLegAreDropped(t *testing.T) {
	t.Parallel()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if body.Model == "model-a" {
			writeFakeResponse(t, w,
				`data: {"choices":[{"index":0,"delta":{"reasoning_items":["rsn_FROM_A"]}}]}`+"\n\n",
				`data: {"error":{"message":"upstream reset"}}`+"\n\n",
			)
			return
		}
		writeFakeResponse(t, w,
			`data: {"choices":[{"index":0,"delta":{"reasoning_items":["rsn_FROM_B"]}}]}`+"\n\n",
			`data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}`+"\n\n",
			"data: [DONE]\n\n",
		)
	}))
	defer srv.Close()

	msg := requireAnswered(t, drain(Stream(context.Background(), RequestFromMessages([]Message{User("hi")}),
		WithModel("openai/model-a,openai/model-b"), withTestProvider(srv.URL+"/", "k"))))
	assert.Equal(t, "openai/model-b", msg.Model)
	assert.Equal(t, []string{"rsn_FROM_B"}, msg.ReasoningItems)
}
