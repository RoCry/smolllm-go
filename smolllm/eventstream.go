package smolllm

import (
	"context"
	"sync"
)

// EventKind discriminates an Event.
type EventKind string

const (
	// EventStart is emitted once, before the first leg is attempted.
	EventStart EventKind = "start"
	// EventTextDelta carries a fragment of answer text.
	EventTextDelta EventKind = "text_delta"
	// EventReasoningDelta carries a fragment of thinking text.
	EventReasoningDelta EventKind = "reasoning_delta"
	// EventToolCallStart announces a new tool call slot.
	EventToolCallStart EventKind = "tool_call_start"
	// EventToolCallDelta carries raw argument JSON text for a tool call.
	EventToolCallDelta EventKind = "tool_call_delta"
	// EventToolCallEnd carries the completed tool call.
	EventToolCallEnd EventKind = "tool_call_end"
	// EventLegFailed reports that the chain is advancing past a failed leg.
	EventLegFailed EventKind = "leg_failed"
	// EventDone is terminal: the call produced an answer.
	EventDone EventKind = "done"
	// EventError is terminal: the call failed or was aborted.
	EventError EventKind = "error"
)

// Terminal reports whether the kind ends a stream.
func (k EventKind) Terminal() bool {
	return k == EventDone || k == EventError
}

// Event is one streamed update. Message is always set and is an immutable
// snapshot: it is never mutated after the Event is delivered.
type Event struct {
	Kind     EventKind
	Delta    string // text, reasoning or tool-argument fragment
	Index    int    // tool-call slot for the EventToolCall* kinds
	ToolCall *ToolCall
	Attempt  *Attempt // set on EventLegFailed
	Message  *AssistantMessage
}

// EventStream is an in-flight call. Events and Result are independent: a caller
// may take Result without draining Events, and vice versa.
//
// Events are buffered in an unbounded queue, so the call never stalls waiting
// for a consumer. A caller that takes Result without draining Events should
// Close the stream to release the goroutine that feeds the channel.
type EventStream struct {
	events chan Event
	notify chan struct{}

	mu          sync.Mutex
	queue       []Event
	inputClosed bool

	cancel     context.CancelFunc
	done       chan struct{}
	abandon    chan struct{}
	closeOnce  sync.Once
	finishOnce sync.Once
	result     *AssistantMessage
}

func newEventStream(cancel context.CancelFunc) *EventStream {
	stream := &EventStream{
		events:      make(chan Event),
		notify:      make(chan struct{}, 1),
		mu:          sync.Mutex{},
		queue:       nil,
		inputClosed: false,
		cancel:      cancel,
		done:        make(chan struct{}),
		abandon:     make(chan struct{}),
		closeOnce:   sync.Once{},
		finishOnce:  sync.Once{},
		result:      nil,
	}
	go stream.pump()
	return stream
}

// Events returns the event channel. It closes once the terminal event has been
// delivered, or once the stream is closed.
func (s *EventStream) Events() <-chan Event {
	return s.events
}

// Result blocks until the call ends and returns the terminal AssistantMessage.
// It never returns nil and never returns an error. It is idempotent.
func (s *EventStream) Result() *AssistantMessage {
	<-s.done
	return s.result
}

// Close aborts the call. Result then reports StopReasonAborted unless the call
// had already finished. It is safe to call more than once.
func (s *EventStream) Close() {
	s.closeOnce.Do(func() {
		s.cancel()
		close(s.abandon)
	})
}

// push queues an event. It never blocks, so a slow or absent consumer can never
// stall the call producing the events.
func (s *EventStream) push(event Event) {
	s.mu.Lock()
	if s.inputClosed {
		s.mu.Unlock()
		return
	}
	s.queue = append(s.queue, event)
	s.mu.Unlock()
	s.wake()
}

// closeInput tells the pump that no further events will be queued.
func (s *EventStream) closeInput() {
	s.mu.Lock()
	s.inputClosed = true
	s.mu.Unlock()
	s.wake()
}

func (s *EventStream) wake() {
	select {
	case s.notify <- struct{}{}:
	default:
	}
}

// pump moves queued events onto the channel, so producers stay unblocked while
// the consumer reads at its own pace.
func (s *EventStream) pump() {
	defer close(s.events)
	for {
		s.mu.Lock()
		if len(s.queue) == 0 {
			finished := s.inputClosed
			s.mu.Unlock()
			if finished {
				return
			}
			select {
			case <-s.notify:
			case <-s.abandon:
				return
			}
			continue
		}
		event := s.queue[0]
		s.queue = s.queue[1:]
		s.mu.Unlock()

		select {
		case s.events <- event:
		case <-s.abandon:
			return
		}
	}
}

// finish latches the terminal message, emits the terminal event and closes the
// stream. Only the first call has any effect.
func (s *EventStream) finish(message *AssistantMessage) {
	s.finishOnce.Do(func() {
		kind := EventDone
		if message.StopReason == StopReasonError || message.StopReason == StopReasonAborted {
			kind = EventError
		}
		s.result = message
		s.push(newEvent(kind, message))
		s.closeInput()
		close(s.done)
	})
}

// newEvent builds an event carrying only a snapshot, which is the shape of every
// event that has no delta or payload of its own.
func newEvent(kind EventKind, message *AssistantMessage) Event {
	return Event{
		Kind:     kind,
		Delta:    "",
		Index:    0,
		ToolCall: nil,
		Attempt:  nil,
		Message:  message,
	}
}
