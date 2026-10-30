package smolllm

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

// 400 bodies captured from the live smolayer (67f8bec) on 2026-10-30.
const (
	// A WorkBuddy leg asked for response_format json_object.
	smolayerFormatRefusal = `{"error":{"message":"workbuddy does not promise JSON schema output",` +
		`"type":"invalid_request_error","param":"response_format","code":null}}`
	// A qoder leg while smolayer holds no qoder login.
	smolayerQoderLoggedOut = "{\"error\":{\"message\":\"unknown provider `qoder`; known providers: codex, " +
		"antigravity, workbuddy, workbuddy-global; `qoder` is not logged in (run `smolayer login qoder`)\"," +
		`"type":"invalid_request_error","param":"model","code":null}}`
	// A request with no messages: wrong for every provider.
	smolayerNoMessages = `{"error":{"message":"a turn needs at least one user, assistant or tool message",` +
		`"type":"invalid_request_error","param":"messages","code":null}}`
)

// OpenAI's answer to max_tokens on a reasoning model, as its docs and SDK issues quote it.
const openAIUnsupportedParameter = `{"error":{"message":"Unsupported parameter: 'max_tokens' is not ` +
	`supported with this model. Use 'max_completion_tokens' instead.","type":"invalid_request_error",` +
	`"param":"max_tokens","code":"unsupported_parameter"}}`

func TestClassifyBadRequestBodies(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		body string
		want Disposition
	}{
		{"smolayer response_format refusal", smolayerFormatRefusal, DispositionAdvance},
		{"smolayer qoder not logged in", smolayerQoderLoggedOut, DispositionAdvance},
		{"openai unsupported parameter", openAIUnsupportedParameter, DispositionAdvance},
		{"smolayer missing messages", smolayerNoMessages, DispositionAbort},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			for _, status := range []int{http.StatusBadRequest, http.StatusUnprocessableEntity} {
				err := &HTTPError{StatusCode: status, Body: tt.body, NoRetry: false}
				assert.Equal(t, tt.want, Classify(err), "status %d", status)
			}
		})
	}
}

// A leg-local 400 hands the call to the next leg; a caller-fault 400 never
// reaches it.
func TestAskBadRequestFailover(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name         string
		body         string
		wantFallback int32
	}{
		{"response_format refusal advances", smolayerFormatRefusal, 1},
		{"logged-out provider advances", smolayerQoderLoggedOut, 1},
		{"missing messages aborts", smolayerNoMessages, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			srv, _, fallback := fallbackServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusBadRequest)
				writeFakeResponse(t, w, tt.body)
			})

			msg := Ask(context.Background(), RequestFromString("hi"),
				WithModel("openai/model-a,gemini/model-b"),
				withTestProvider(srv.URL+"/", "test-key"),
			)
			assert.Equal(t, tt.wantFallback, fallback.Load())
			if tt.wantFallback == 0 {
				requireFailed(t, msg)
				return
			}
			requireAnswered(t, msg)
			assert.Equal(t, "fallback answer", msg.Content)
		})
	}
}
