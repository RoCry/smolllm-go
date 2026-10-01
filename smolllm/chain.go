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
//
// A leg whose answer text or tool-call fragments have been emitted commits the
// call: if it then fails, the stream ends in that error rather than advancing,
// because a consumer forwarding events would splice two answers together.
// Reasoning alone does not commit. Use Ask to fall back after partial output.
func (c *Client) Stream(ctx context.Context, req Request, opts ...Option) *EventStream {
	return c.start(ctx, req, true, opts...)
}

// Ask drains a stream and returns its terminal AssistantMessage. It never
// returns nil: check StopReason to tell an answer from a failure. Nothing
// reaches the caller before the turn ends, so a leg that fails after partial
// output still falls back, and only the winner's answer is returned.
func (c *Client) Ask(ctx context.Context, req Request, opts ...Option) *AssistantMessage {
	stream := c.start(ctx, req, false, opts...)
	// Draining rather than only taking Result lets the pump goroutine finish on
	// its own instead of parking until Close.
	for range stream.Events() {
	}
	return stream.Result()
}

// start runs the chain on its own goroutine. commitOnOutput is whether emitted
// answer output pins the chain to its leg; see Stream.
func (c *Client) start(ctx context.Context, req Request, commitOnOutput bool, opts ...Option) *EventStream {
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
		c.runChain(callCtx, req, options, commitOnOutput, stream)
	}()

	return stream
}

// chainState accumulates one call. It is the streamSink the parser writes into,
// so text and tool fragments become events as they arrive.
type chainState struct {
	acc    *messageAccumulator
	stream *EventStream
	// commitOnOutput makes a failure after emitted answer output terminal.
	commitOnOutput bool
	// emitted records that the current attempt has emitted answer text or a
	// tool-call fragment. Reasoning does not count.
	emitted bool
}

func (s *chainState) text(fragment delta) {
	if fragment.Content != "" {
		s.emitted = true
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
		s.emitted = true
		s.emit(Event{
			Kind: EventToolCallStart, Delta: "", Index: fragment.Index,
			ToolCall: nil, Attempt: nil, Message: nil,
		})
	}
	if fragment.Arguments != "" {
		s.emitted = true
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
func (c *Client) runChain(
	ctx context.Context, req Request, opts Options, commitOnOutput bool, stream *EventStream,
) {
	state := &chainState{
		acc: newMessageAccumulator(), stream: stream, commitOnOutput: commitOnOutput, emitted: false,
	}

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
	var legDeadline time.Time
	if opts.LegBudget > 0 {
		legDeadline = time.Now().Add(opts.LegBudget)
	}
	for retry := range opts.MaxRetries {
		if retry > 0 {
			if !waitBeforeRetry(ctx, opts, model, retry) {
				return false, DispositionAbort
			}
			if !legDeadline.IsZero() && !time.Now().Before(legDeadline) {
				opts.Logger.Warn("leg budget spent; advancing", "model", model, "budget", opts.LegBudget)
				return false, DispositionAdvance
			}
		}

		// A leg starts from an empty turn: text a failed leg produced must not
		// bleed into the next candidate's answer.
		state.acc.resetLeg()
		state.emitted = false

		attempt, legErr := c.attemptLeg(ctx, req, opts, model, retry, legDeadline, state)
		attempt.Emitted = state.emitted
		if legErr != nil && state.commitOnOutput && state.emitted {
			commitFailure(legErr)
		}
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
			"emitted", attempt.Emitted,
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

// commitFailure stops the chain at a leg whose answer output a Stream consumer
// has already received: a retry or the next leg would splice a second answer
// onto the first. A failure that aborts anyway keeps its own reason.
func commitFailure(leg *LegError) {
	if leg.Disposition == DispositionAbort {
		return
	}
	leg.Disposition = DispositionAbort
	leg.Err = fmt.Errorf("%w (output already emitted, so the chain did not advance)", leg.Err)
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
	ctx context.Context, req Request, opts Options, model string, retry int, legDeadline time.Time,
	state *chainState,
) (Attempt, *LegError) {
	exec, err := newCallExecution(ctx, req, opts, model, c.balancer)
	if err != nil {
		// The leg could not even be prepared: leg-local, so the chain advances.
		leg := newLegError(nil, model, retry, err)
		return failedAttempt(nil, retry, leg, time.Time{}, nil), leg
	}
	defer exec.cancel(nil)

	// Snapshots from here on, the failure ones included, name the leg being
	// tried; resetLeg cleared the previous leg's identity.
	state.acc.provider = exec.call.Provider.Name
	state.acc.model = exec.call.Model
	state.acc.modelName = exec.call.ModelName

	fail := func(cause error, outcome *legOutcome) (Attempt, *LegError) {
		leg := newLegError(exec.call, model, retry, cause)
		return failedAttempt(exec.call, retry, leg, exec.start, outcome), leg
	}

	// The budget bounds only the wait for a response to start: once a leg is
	// streaming it is alive, and the whole-call deadline bounds the rest.
	disarm := exec.armLegBudget(legDeadline)

	resp, err := exec.do("sending request")
	if err != nil {
		disarm()
		return fail(exec.budgetCause(err, opts.LegBudget), nil)
	}
	// Late-bound on purpose: a stream_options retry rebinds resp, and the body
	// that must be closed here is whichever response is current. The superseded
	// body is already closed inside retryWithoutStreamUsage.
	defer func() { resp.Body.Close() }()

	if resp.StatusCode >= http.StatusBadRequest {
		retryResp, retried, retryErr := exec.retryWithoutStreamUsage(resp)
		if retryErr != nil {
			disarm()
			return fail(exec.budgetCause(retryErr, opts.LegBudget), nil)
		}
		if retried {
			resp = retryResp
		}
	}
	if resp.StatusCode >= http.StatusBadRequest {
		statusErr := httpError(resp)
		disarm()
		return fail(exec.budgetCause(statusErr, opts.LegBudget), nil)
	}
	if disarm() {
		return fail(exec.budgetCause(context.Canceled, opts.LegBudget), nil)
	}

	outcome := consumeLegStream(exec.requestContext(), opts.Logger, resp.Body, exec.start, state)
	state.acc.finishReason = outcome.finishReason
	if outcome.err != nil {
		return fail(outcome.err, &outcome)
	}

	usage := outcome.usage.usage
	if !outcome.usage.reported {
		usage = estimateUsage(exec.call.InputTokens, state.acc.content.String()+state.acc.reasoning.String())
	}
	state.acc.usage = usage

	if guardErr := guardResponse(state, opts, exec.call, outcome, usage); guardErr != nil {
		return fail(guardErr, &outcome)
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
// to a caller. Each guard fails the leg so the chain can route around it; tool
// calls truncated at a declared max_tokens are the one exception, passed on
// with a warning.
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

	// Tool calls cut off mid-argument cannot run: the argument JSON is
	// incomplete. Without a declared max_tokens the cut came from a provider or
	// relay default, and the next leg's default may be larger, so the leg fails.
	// A cap the caller declared cuts every leg at the same place, so the calls
	// go back with stop reason length for the caller to handle.
	if outcome.finishReason == finishReasonLength && len(outcome.toolCalls) > 0 {
		calls := describeToolCalls(outcome.toolCalls)
		if opts.MaxTokens == nil {
			return fmt.Errorf("model %q truncated mid tool call (finish_reason=length): %s", call.Model, calls)
		}
		opts.Logger.Warn("truncated mid tool call at the declared max_tokens",
			"model", call.Model, "max_tokens", *opts.MaxTokens, "tool_calls", calls)
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
