package smolllm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Stream starts a call on the shared client and returns synchronously.
func Stream(ctx context.Context, req Request, opts ...Option) *EventStream {
	return sharedClient().Stream(ctx, req, opts...)
}

// Ask drains a stream on the shared client and returns its terminal message.
func Ask(ctx context.Context, req Request, opts ...Option) *AssistantMessage {
	return sharedClient().Ask(ctx, req, opts...)
}

// Stream starts a call and returns synchronously. It NEVER reports an
// operational failure as a Go error: the terminal AssistantMessage carries the
// StopReason and, when something went wrong, the ErrorMessage. Programmer errors
// panic.
func (c *Client) Stream(ctx context.Context, req Request, opts ...Option) *EventStream {
	if ctx == nil {
		panic("smolllm: context must not be nil")
	}

	options := c.callOptions(opts...)
	// One deadline bounds the whole call: every leg, every retry, the backoff
	// waits between them, and the consumption of the stream.
	callCtx, cancel := deriveContext(ctx, options.Timeout)
	stream := newEventStream(cancel)

	go func() {
		defer cancel()
		c.runChain(callCtx, req, options, stream)
	}()

	return stream
}

// Ask drains a stream and returns its terminal AssistantMessage. It never
// returns nil: check StopReason to tell an answer from a failure.
func (c *Client) Ask(ctx context.Context, req Request, opts ...Option) *AssistantMessage {
	stream := c.Stream(ctx, req, opts...)
	// Draining rather than only taking Result lets the pump goroutine finish on
	// its own instead of parking until Close.
	for range stream.Events() { //nolint:revive // draining is the point
	}
	return stream.Result()
}

// chainState accumulates one call. It is the streamSink the parser writes into,
// so text and tool fragments become events as they arrive.
type chainState struct {
	acc    *messageAccumulator
	stream *EventStream
}

func (s *chainState) text(fragment delta) {
	if fragment.Content != "" {
		s.acc.content.WriteString(fragment.Content)
		s.emit(Event{
			Kind: EventTextDelta, Delta: fragment.Content, Index: 0,
			ToolCall: nil, Attempt: nil, Message: nil,
		})
	}
	if fragment.Reasoning != "" {
		s.acc.reasoning.WriteString(fragment.Reasoning)
		s.emit(Event{
			Kind: EventReasoningDelta, Delta: fragment.Reasoning, Index: 0,
			ToolCall: nil, Attempt: nil, Message: nil,
		})
	}
}

func (s *chainState) toolAccumulator() *toolCallAccumulator {
	return s.acc.tools
}

func (s *chainState) toolFragment(fragment toolCallFragment) {
	if fragment.Started {
		s.emit(Event{
			Kind: EventToolCallStart, Delta: "", Index: fragment.Index,
			ToolCall: nil, Attempt: nil, Message: nil,
		})
	}
	if fragment.Arguments != "" {
		s.emit(Event{
			Kind: EventToolCallDelta, Delta: fragment.Arguments, Index: fragment.Index,
			ToolCall: nil, Attempt: nil, Message: nil,
		})
	}
}

// emit attaches a fresh snapshot and queues the event. Every consumer therefore
// sees an AssistantMessage that later accumulation cannot change.
func (s *chainState) emit(event Event) {
	event.Message = s.acc.snapshot()
	s.stream.push(event)
}

// runChain drives the fallback chain to a terminal message. It always finishes
// the stream, whatever happens.
func (c *Client) runChain(ctx context.Context, req Request, opts Options, stream *EventStream) {
	state := &chainState{acc: newMessageAccumulator(), stream: stream}

	// Result must never block forever. Every path below latches a terminal
	// message, and finish only honours the first call, so this is a no-op unless
	// a path is ever added that forgets to.
	defer finishFailed(state, terminalReason(ctx), "chain ended without a result")

	// A malformed request is data, not a coding contract violation, so it ends
	// the call as a terminal error rather than a panic.
	if err := req.Validate(); err != nil {
		finishFailed(state, StopReasonError, err.Error())
		return
	}

	selector, err := createSelector(opts)
	if err != nil {
		finishFailed(state, StopReasonError, err.Error())
		return
	}

	state.emit(newEvent(EventStart, nil))

	var legErrors []error
	for {
		model, ok := selector.NextModel()
		if !ok {
			break
		}

		done, disposition := c.runLeg(ctx, req, opts, model, state, &legErrors)
		if done {
			return
		}
		if disposition == DispositionAbort {
			break
		}
	}

	finishFailed(state, terminalReason(ctx), joinLegErrors(legErrors).Error())
}

// runLeg attempts one model, retrying it while the classifier says to. It
// reports whether the call finished, and otherwise what to do next.
func (c *Client) runLeg(
	ctx context.Context, req Request, opts Options, model string, state *chainState, legErrors *[]error,
) (done bool, disposition Disposition) {
	for retry := range opts.MaxRetries {
		if retry > 0 {
			if !waitBeforeRetry(ctx, opts, model, retry) {
				return false, DispositionAbort
			}
		}

		// A leg starts from an empty turn: text a failed leg produced must not
		// bleed into the next candidate's answer.
		state.acc.resetLeg()

		attempt, legErr := c.attemptLeg(ctx, req, opts, model, retry, state)
		state.acc.attempts = append(state.acc.attempts, attempt)
		if opts.Hook != nil {
			opts.Hook(attempt)
		}

		if legErr == nil {
			finishSucceeded(state)
			return true, DispositionAdvance
		}

		*legErrors = append(*legErrors, legErr)
		opts.Logger.Warn("leg failed",
			"model", model,
			"retry", retry,
			"disposition", legErr.Disposition.String(),
			"error", legErr.Error(),
		)

		recorded := attempt
		state.emit(Event{
			Kind: EventLegFailed, Delta: "", Index: 0,
			ToolCall: nil, Attempt: &recorded, Message: nil,
		})

		switch legErr.Disposition {
		case DispositionRetry:
			continue
		case DispositionAbort:
			return false, DispositionAbort
		case DispositionAdvance:
			return false, DispositionAdvance
		default:
			return false, DispositionAdvance
		}
	}
	// Retries exhausted: the model is not going to recover, so move on.
	return false, DispositionAdvance
}

// waitBeforeRetry sleeps out the backoff. It reports false when the call ended
// while waiting.
func waitBeforeRetry(ctx context.Context, opts Options, model string, retry int) bool {
	wait := retryDelay(retry - 1)
	opts.Logger.Warn("retrying after transient error", "model", model, "attempt", retry+1, "delay", wait)
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// attemptLeg runs one HTTP attempt and accumulates whatever it produced. A nil
// LegError means the answer is usable.
func (c *Client) attemptLeg(
	ctx context.Context, req Request, opts Options, model string, retry int, state *chainState,
) (Attempt, *LegError) {
	exec, err := newCallExecution(ctx, req, opts, model, c.balancer)
	if err != nil {
		// The leg could not even be prepared: leg-local, so the chain advances.
		leg := newLegError(nil, model, retry, err)
		return failedAttempt(nil, retry, leg, time.Time{}, nil), leg
	}
	defer exec.cancel()

	fail := func(cause error, reported *reportedUsage) (Attempt, *LegError) {
		leg := newLegError(exec.call, model, retry, cause)
		return failedAttempt(exec.call, retry, leg, exec.start, reported), leg
	}

	resp, err := exec.do("sending request")
	if err != nil {
		return fail(err, nil)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= http.StatusBadRequest {
		retryResp, retried, retryErr := exec.retryWithoutStreamUsage(resp)
		if retryErr != nil {
			return fail(retryErr, nil)
		}
		if retried {
			resp = retryResp
			defer func() { _ = resp.Body.Close() }()
		}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return fail(httpError(resp), nil)
	}

	// Identity is known now, so snapshots taken during the stream name the leg
	// that is actually answering.
	state.acc.provider = exec.call.Provider.Name
	state.acc.model = exec.call.Model
	state.acc.modelName = exec.call.ModelName

	outcome := consumeLegStream(exec.requestContext(), opts.Logger, resp.Body, exec.start, state)
	state.acc.finishReason = outcome.finishReason
	if outcome.err != nil {
		return fail(outcome.err, &outcome.usage)
	}

	usage := outcome.usage.usage
	if !outcome.usage.reported {
		usage = estimateUsage(exec.call.InputTokens, state.acc.content.String()+state.acc.reasoning.String())
	}
	state.acc.usage = usage

	if guardErr := guardResponse(state, opts, exec.call, outcome, usage); guardErr != nil {
		return fail(guardErr, &outcome.usage)
	}

	// The complete call is only announced once the leg is accepted: a leg the
	// guards reject must not look like it produced usable calls.
	for position, index := range outcome.toolIndexes {
		completed := outcome.toolCalls[position]
		state.emit(Event{
			Kind: EventToolCallEnd, Delta: "", Index: index,
			ToolCall: &completed, Attempt: nil, Message: nil,
		})
	}

	// Backtick stripping rewrites the finished answer, so it happens once the
	// whole turn is in hand rather than per fragment.
	if opts.RemoveBackticks {
		stripped := stripBackticks(state.acc.content.String())
		state.acc.content.Reset()
		state.acc.content.WriteString(stripped)
	}

	total := time.Since(exec.start)
	opts.Logger.Info(
		formatMetrics(exec.call.ModelName, usage.Input, usage.Output, total, outcome.ttft),
		"model", exec.call.ModelName,
	)

	attempt := newAttempt(exec.call, retry)
	attempt.Usage = usage
	attempt.Duration = total
	attempt.TTFT = outcome.ttft
	return attempt, nil
}

// guardResponse rejects an answer that arrived intact on the wire but is unusable
// to a caller. Each guard fails the leg so the chain can route around it.
func guardResponse(
	state *chainState, opts Options, call *preparedCall, outcome legOutcome, usage Usage,
) error {
	content := state.acc.content.String()
	reasoning := state.acc.reasoning.String()
	trimmed := strings.TrimSpace(content)

	// An assistant turn that only requests tool calls carries no text, but it is
	// a complete, useful response.
	if trimmed == "" && strings.TrimSpace(reasoning) == "" && len(outcome.toolCalls) == 0 {
		return fmt.Errorf("model %q returned empty response", call.Model)
	}

	// Reasoning spent the whole output budget before any answer was emitted. An
	// empty completion is useless to every caller. Truncated-but-non-empty
	// content still passes: partial output may be usable and retrying elsewhere
	// would double-spend tokens.
	if outcome.finishReason == finishReasonLength && trimmed == "" && len(outcome.toolCalls) == 0 {
		return fmt.Errorf(
			"model %q truncated before any content (finish_reason=length, reasoning consumed the output budget)",
			call.Model,
		)
	}

	// Tool calls cut off mid-argument are unusable: the argument JSON no longer
	// parses. Fail the leg rather than hand back a broken call.
	if outcome.finishReason == finishReasonLength && len(outcome.toolCalls) > 0 {
		return fmt.Errorf("model %q truncated mid tool call (finish_reason=length)", call.Model)
	}

	// Suspiciously short output usually means context window overflow. A
	// tool-call-only turn is legitimately tiny, so it is exempt.
	if opts.MinOutputTokens > 0 && len(outcome.toolCalls) == 0 &&
		usage.Output < opts.MinOutputTokens && call.InputTokens > 1000 {
		return fmt.Errorf(
			"model %q returned suspiciously short response (%d output tokens for %d input tokens, min=%d)",
			call.Model, usage.Output, call.InputTokens, opts.MinOutputTokens,
		)
	}
	return nil
}

// finishSucceeded latches the winning leg's turn as the terminal message.
func finishSucceeded(state *chainState) {
	acc := state.acc
	acc.stopReason = stopReasonFor(acc.finishReason, acc.tools.result())
	acc.errorMessage = ""
	state.stream.finish(acc.snapshot())
}

// finishFailed latches a failure as the terminal message. The chain never
// reports an operational failure as a Go error.
func finishFailed(state *chainState, reason StopReason, message string) {
	state.acc.stopReason = reason
	state.acc.errorMessage = message
	state.stream.finish(state.acc.snapshot())
}

// terminalReason distinguishes a caller's cancellation from an expired deadline:
// only the former is an abort.
func terminalReason(ctx context.Context) StopReason {
	if errors.Is(ctx.Err(), context.Canceled) {
		return StopReasonAborted
	}
	return StopReasonError
}
