package smolllm

import (
	"strings"
	"time"
)

// StopReason says why generation ended, normalized across providers.
type StopReason string

const (
	// StopReasonPending marks a non-terminal snapshot: the call is still running.
	StopReasonPending StopReason = "pending"
	// StopReasonStop means the model finished its turn.
	StopReasonStop StopReason = "stop"
	// StopReasonLength means output was truncated by a token limit.
	StopReasonLength StopReason = "length"
	// StopReasonToolUse means the model asked for one or more tool calls.
	StopReasonToolUse StopReason = "tool_use"
	// StopReasonError means every leg of the chain failed.
	StopReasonError StopReason = "error"
	// StopReasonAborted means the caller cancelled the call or closed the stream.
	StopReasonAborted StopReason = "aborted"
)

// Terminal reports whether the reason ends a stream.
func (r StopReason) Terminal() bool {
	return r != StopReasonPending
}

// stopReasonFor normalizes a provider finish reason. Length wins: calls cut off
// by the output cap must never look runnable. Otherwise a turn carrying tool
// calls is tool use however the provider labelled it: Gemini reports "stop"
// alongside tool calls, so the calls themselves are the stronger signal.
func stopReasonFor(finishReason string, toolCalls []ToolCall) StopReason {
	if finishReason == finishReasonLength {
		return StopReasonLength
	}
	if finishReason == finishReasonToolCalls || len(toolCalls) > 0 {
		return StopReasonToolUse
	}
	return StopReasonStop
}

const (
	finishReasonLength    = "length"
	finishReasonToolCalls = "tool_calls"
)

// AssistantMessage accumulates one assistant turn. Every Event carries a
// snapshot of it, and the terminal snapshot is what Result returns.
type AssistantMessage struct {
	Content   string     `json:"content"`
	Reasoning string     `json:"reasoning"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	StopReason StopReason `json:"stop_reason"`
	// ErrorMessage joins every failed leg, so a reader learns why the whole
	// chain failed rather than only its last candidate.
	ErrorMessage string `json:"error_message,omitempty"`
	// FinishReason is the winning leg's provider string, verbatim and never
	// normalized. Use StopReason for a normalized answer.
	FinishReason string `json:"finish_reason"`

	Provider  string `json:"provider"`   // empty for a bare model spec
	Model     string `json:"model"`      // full spec, e.g. "deepseek/deepseek-v4-flash"
	ModelName string `json:"model_name"` // wire model, e.g. "deepseek-v4-flash"

	Usage    Usage     `json:"usage"`              // the winning leg only
	Attempts []Attempt `json:"attempts,omitempty"` // every leg tried, in order

	Started  time.Time     `json:"started"`
	Duration time.Duration `json:"duration"`
}

// messageAccumulator builds an AssistantMessage as a call runs. Text is
// accumulated in builders so a snapshot costs no copy of the text itself.
type messageAccumulator struct {
	content   strings.Builder
	reasoning strings.Builder
	tools     *toolCallAccumulator

	stopReason   StopReason
	errorMessage string
	finishReason string

	provider  string
	modelName string
	model     string

	usage    Usage
	attempts []Attempt
	started  time.Time
}

func newMessageAccumulator() *messageAccumulator {
	return &messageAccumulator{
		content:      strings.Builder{},
		reasoning:    strings.Builder{},
		tools:        newToolCallAccumulator(),
		stopReason:   StopReasonPending,
		errorMessage: "",
		finishReason: "",
		provider:     "",
		modelName:    "",
		model:        "",
		usage:        newUsage(0, 0, 0, 0, true),
		attempts:     nil,
		started:      time.Now().UTC(),
	}
}

// resetLeg clears the text a failed leg produced, so the next leg starts from an
// empty turn rather than appending to a partial answer.
func (a *messageAccumulator) resetLeg() {
	a.content.Reset()
	a.reasoning.Reset()
	a.tools = newToolCallAccumulator()
	a.finishReason = ""
	a.usage = newUsage(0, 0, 0, 0, true)
}

// snapshot returns an immutable view of the turn so far. The caller may hold it
// indefinitely: later accumulation never changes the bytes it can see.
func (a *messageAccumulator) snapshot() *AssistantMessage {
	message := &AssistantMessage{
		Content:      a.content.String(),
		Reasoning:    a.reasoning.String(),
		ToolCalls:    a.tools.snapshot(),
		StopReason:   a.stopReason,
		ErrorMessage: a.errorMessage,
		FinishReason: a.finishReason,
		Provider:     a.provider,
		Model:        a.model,
		ModelName:    a.modelName,
		Usage:        a.usage,
		Attempts:     nil,
		Started:      a.started,
		Duration:     time.Since(a.started),
	}
	if len(a.attempts) > 0 {
		message.Attempts = make([]Attempt, len(a.attempts))
		copy(message.Attempts, a.attempts)
	}
	return message
}

// Usage counts tokens for one call. Input EXCLUDES cached tokens, which are
// reported separately as CacheRead; Output INCLUDES Reasoning. Estimated marks
// counts derived by heuristic because the provider sent no usage frame.
//
// Usage stops at tokens: pricing is the caller's concern, so there are no cost
// fields and there never will be.
type Usage struct {
	Input     int  `json:"input"`
	Output    int  `json:"output"`
	CacheRead int  `json:"cache_read"`
	Reasoning int  `json:"reasoning"`
	Total     int  `json:"total"`
	Estimated bool `json:"estimated"`
}

// newUsage builds a Usage with Total derived from its parts, so no caller can
// report a total that disagrees with them.
func newUsage(input, output, cacheRead, reasoning int, estimated bool) Usage {
	return Usage{
		Input:     input,
		Output:    output,
		CacheRead: cacheRead,
		Reasoning: reasoning,
		Total:     input + output + cacheRead,
		Estimated: estimated,
	}
}

// Attempt records one leg of the fallback chain, successful or not. It is both
// the payload of the request hook and, once the chain finishes, an element of
// the completed call's attempt list.
type Attempt struct {
	Provider   string        `json:"provider"`
	Model      string        `json:"model"`
	ModelName  string        `json:"model_name"`
	APIKeyHint string        `json:"api_key_hint"`
	Retry      int           `json:"retry"` // 0 on the first try of this leg
	Usage      Usage         `json:"usage"`
	Duration   time.Duration `json:"duration"`
	TTFT       time.Duration `json:"ttft"` // -1 when no token arrived
	Err        *LegError     `json:"error,omitempty"`
}

// Failed reports whether this attempt ended in failure. Prefer it to comparing
// Err against nil in an error-typed variable, which a typed nil would defeat.
func (a Attempt) Failed() bool { return a.Err != nil }

// newAttempt builds an Attempt carrying only routing identity. The chain fills
// in usage, timings and any error as the leg runs.
func newAttempt(call *preparedCall, retry int) Attempt {
	attempt := Attempt{
		Provider:   "",
		Model:      "",
		ModelName:  "",
		APIKeyHint: "",
		Retry:      retry,
		Usage:      newUsage(0, 0, 0, 0, true),
		Duration:   0,
		TTFT:       -1,
		Err:        nil,
	}
	if call != nil {
		attempt.Provider = call.Provider.Name
		attempt.Model = call.Model
		attempt.ModelName = call.ModelName
		attempt.APIKeyHint = previewAPIKey(call.APIKey)
		attempt.Usage = newUsage(call.InputTokens, 0, 0, 0, true)
	}
	return attempt
}
