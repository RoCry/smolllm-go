package smolllm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// legRecorder is a fake provider that keeps the decoded request body of every
// leg, keyed by the wire model name, so a test can assert what each leg sent
// rather than only what the winning one did.
type legRecorder struct {
	mu       sync.Mutex
	payloads map[string]map[string]any
}

func newLegRecorder() *legRecorder {
	return &legRecorder{mu: sync.Mutex{}, payloads: map[string]map[string]any{}}
}

// record decodes one request body and returns the wire model name it carried.
func (r *legRecorder) record(t *testing.T, body io.Reader) string {
	t.Helper()

	var payload map[string]any
	if err := json.NewDecoder(body).Decode(&payload); err != nil {
		t.Errorf("decode fake provider request: %v", err)
		return ""
	}
	model, ok := payload["model"].(string)
	if !ok {
		t.Errorf("fake provider request carries no string model: %v", payload["model"])
		return ""
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.payloads[model] = payload
	return model
}

func (r *legRecorder) payload(t *testing.T, model string) map[string]any {
	t.Helper()

	r.mu.Lock()
	defer r.mu.Unlock()
	payload, ok := r.payloads[model]
	require.True(t, ok, "leg %q never sent a request", model)
	return payload
}

// newTwoLegServer fails model-a and answers on model-b, so one call exercises
// both legs of a two-model chain.
func newTwoLegServer(t *testing.T, rec *legRecorder) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch rec.record(t, r.Body) {
		case testModelA:
			http.Error(w, "upstream unavailable", http.StatusInternalServerError)
		case testModelB:
			writeChatSuccess(t, w, "fallback answer", "stop")
		default:
			http.Error(w, "unexpected model", http.StatusBadRequest)
		}
	}))
}

func TestLegReasoningEffortOverridesOnlyItsOwnLeg(t *testing.T) {
	t.Parallel()

	rec := newLegRecorder()
	srv := newTwoLegServer(t, rec)
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,openai/model-b"),
		withTestProvider(srv.URL+"/", testAPIKey),
		WithReasoningEffort("high"),
		WithLegReasoningEffort("openai/model-a", "none"),
		// One attempt per leg: this test is about what each leg sends, not about
		// how long the chain is willing to wait for the first one.
		WithMaxRetries(1),
	)
	requireAnswered(t, msg)

	assert.Equal(t, "none", rec.payload(t, testModelA)["reasoning_effort"], "the overridden leg")
	assert.Equal(t, "high", rec.payload(t, testModelB)["reasoning_effort"], "the leg without an override")
}

func TestLegReasoningEffortLeavesUnnamedLegsUnset(t *testing.T) {
	t.Parallel()

	rec := newLegRecorder()
	srv := newTwoLegServer(t, rec)
	defer srv.Close()

	// No WithReasoningEffort: the leg with no override of its own has nothing to
	// fall back to, so it must send no reasoning_effort at all.
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/model-a,openai/model-b"),
		withTestProvider(srv.URL+"/", testAPIKey),
		WithLegReasoningEffort("openai/model-a", "none"),
		WithMaxRetries(1),
	)
	requireAnswered(t, msg)

	assert.Equal(t, "none", rec.payload(t, testModelA)["reasoning_effort"])
	assert.NotContains(t, rec.payload(t, testModelB), "reasoning_effort")
}

func TestLegReasoningEffortNormalizesAndLastCallWins(t *testing.T) {
	t.Parallel()

	opts := applyOptions(
		WithLegReasoningEffort(testChatModel, "none"),
		WithLegReasoningEffort("  "+testChatModel+"  ", "  HIGH  "),
	)
	assert.Equal(t, map[string]string{testChatModel: "high"}, opts.LegEfforts)
}

func TestLegReasoningEffortIsCopiedPerCall(t *testing.T) {
	t.Parallel()

	// Options travels by value, so a per-call override must not reach into the
	// map the client was built with.
	base := applyOptions(WithLegReasoningEffort(testChatModel, "none"))
	perCall := base
	WithLegReasoningEffort("openai/other", "high")(&perCall)

	assert.Equal(t, map[string]string{testChatModel: "none"}, base.LegEfforts)
	assert.Len(t, perCall.LegEfforts, 2)
}

func TestReasoningEffortForPrefersTheLegOverride(t *testing.T) {
	t.Parallel()

	global := applyOptions(WithReasoningEffort("high"))
	assert.Equal(t, "high", *global.reasoningEffortFor(testChatModel))
	assert.Nil(t, applyOptions().reasoningEffortFor(testChatModel), "no effort configured at all")

	both := applyOptions(
		WithReasoningEffort("high"),
		WithLegReasoningEffort(testChatModel, "none"),
	)
	assert.Equal(t, "none", *both.reasoningEffortFor(testChatModel), "the leg override wins")
	assert.Equal(t, "high", *both.reasoningEffortFor("openai/other"), "an unnamed leg keeps the global")
}

func TestWithLegReasoningEffortPanicsOnEmpty(t *testing.T) {
	t.Parallel()

	require.PanicsWithValue(t, "WithLegReasoningEffort: model must not be empty", func() {
		WithLegReasoningEffort("  ", "none")
	})
	require.PanicsWithValue(t, "WithLegReasoningEffort: effort must not be empty", func() {
		WithLegReasoningEffort(testChatModel, "")
	})
}

func TestValidateChecksEachLegEffortAgainstItsOwnProvider(t *testing.T) {
	t.Parallel()

	// "minimal" is on the OpenAI allowlist but not Ollama's, so the same value
	// is fine as the chain-wide default and wrong as this leg's override.
	err := Validate(
		WithModel(testChatModel+",ollama/qwen3"),
		withTestProvider("https://example.invalid", testAPIKey),
		WithReasoningEffort("none"),
		WithLegReasoningEffort("ollama/qwen3", "minimal"),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ollama/qwen3")
	assert.Contains(t, err.Error(), "minimal")
	assert.NotContains(t, err.Error(), testChatModel, "the other leg's effort is valid for its provider")
}

func TestValidateAcceptsALegEffortItsProviderAllows(t *testing.T) {
	t.Parallel()

	require.NoError(t, Validate(
		WithModel(testChatModel+",ollama/qwen3"),
		withTestProvider("https://example.invalid", testAPIKey),
		WithReasoningEffort("minimal"),
		// Ollama rejects the chain-wide "minimal", so this leg names its own.
		WithLegReasoningEffort("ollama/qwen3", "low"),
	))
}

func TestValidateRejectsALegEffortForAModelNotInTheChain(t *testing.T) {
	t.Parallel()

	err := Validate(
		WithModel(testChatModel),
		withTestProvider("https://example.invalid", testAPIKey),
		WithLegReasoningEffort("openai/gpt-4o", "none"),
	)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `"openai/gpt-4o"`)
	assert.Contains(t, err.Error(), "not in the chain")
}

func TestValidateAcceptsALegEffortForEveryModelInASet(t *testing.T) {
	t.Parallel()

	// WithModelSet hands the legs out in random order, so the check has to see
	// the whole chain rather than the first leg it happens to draw.
	require.NoError(t, Validate(
		WithModelSet(testChatModel, "openai/gpt-4o"),
		withTestProvider("https://example.invalid", testAPIKey),
		WithLegReasoningEffort("openai/gpt-4o", "none"),
	))
}
