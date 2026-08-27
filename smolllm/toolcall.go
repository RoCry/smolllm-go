package smolllm

import (
	"encoding/json"
	"fmt"
	"sort"
)

// ToolCall is a provider-issued request to run a named function, surfaced
// verbatim. The caller executes it and replays the assistant and tool messages;
// the library runs no agentic loop and never inspects the argument JSON.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
	// Extra carries provider keys the library does not model — Gemini's
	// extra_content.google.thought_signature, for one, which the provider expects
	// echoed back on replay. Preserved so a replayed turn is lossless.
	Extra map[string]json.RawMessage `json:"-"`
}

// ToolCallFunction names the function and carries its arguments.
type ToolCallFunction struct {
	Name string `json:"name"`
	// Arguments is opaque JSON text: never parsed, validated or repaired.
	Arguments string `json:"arguments"`
}

type toolCallWire struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

// MarshalJSON writes the wire fields plus any preserved provider extras.
func (t ToolCall) MarshalJSON() ([]byte, error) {
	encoded, err := json.Marshal(toolCallWire{ID: t.ID, Type: t.Type, Function: t.Function})
	if err != nil {
		return nil, fmt.Errorf("encode tool call: %w", err)
	}
	if len(t.Extra) == 0 {
		return encoded, nil
	}
	var merged map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &merged); err != nil {
		return nil, fmt.Errorf("encode tool call extras: %w", err)
	}
	for key, value := range t.Extra {
		merged[key] = value
	}
	out, err := json.Marshal(merged)
	if err != nil {
		return nil, fmt.Errorf("encode tool call extras: %w", err)
	}
	return out, nil
}

// UnmarshalJSON keeps unknown provider keys in Extra rather than dropping them.
func (t *ToolCall) UnmarshalJSON(data []byte) error {
	var wire toolCallWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return fmt.Errorf("decode tool call: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("decode tool call: %w", err)
	}
	for _, known := range []string{"id", "type", "function"} {
		delete(raw, known)
	}
	t.ID = wire.ID
	t.Type = wire.Type
	t.Function = wire.Function
	t.Extra = nil
	if len(raw) > 0 {
		t.Extra = raw
	}
	return nil
}

// extraFields renders Extra for the OpenAI SDK's param escape hatch.
func (t ToolCall) extraFields() map[string]any {
	if len(t.Extra) == 0 {
		return nil
	}
	fields := make(map[string]any, len(t.Extra))
	for key, value := range t.Extra {
		fields[key] = value
	}
	return fields
}

// toolCallDelta is one streamed fragment of a tool call.
type toolCallDelta struct {
	Index    *int
	ID       string
	Type     string
	Function *toolCallFunctionDelta
	Extra    map[string]json.RawMessage
}

type toolCallFunctionDelta struct {
	Name      string  `json:"name"`
	Arguments *string `json:"arguments"`
}

// UnmarshalJSON keeps unknown keys so provider extras survive accumulation.
func (d *toolCallDelta) UnmarshalJSON(data []byte) error {
	var wire struct {
		Index    *int                   `json:"index"`
		ID       string                 `json:"id"`
		Type     string                 `json:"type"`
		Function *toolCallFunctionDelta `json:"function"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return fmt.Errorf("decode tool call delta: %w", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return fmt.Errorf("decode tool call delta: %w", err)
	}
	// `index` is consumed for slotting: it is a streaming artifact, and the
	// assembled list is ordered by it instead.
	for _, known := range []string{"index", "id", "type", "function"} {
		delete(raw, known)
	}
	d.Index = wire.Index
	d.ID = wire.ID
	d.Type = wire.Type
	d.Function = wire.Function
	d.Extra = nil
	if len(raw) > 0 {
		d.Extra = raw
	}
	return nil
}

// toolCallAccumulator reassembles streamed tool-call deltas into whole calls.
//
// Providers stream a tool call across many frames: the first carries the id and
// function name, later ones append fragments of the argument JSON. Fragments are
// never pushed to stream consumers — a caller can only act on a complete call, so
// the assembled list is exposed once the stream ends.
type toolCallAccumulator struct {
	slots    map[int]*ToolCall
	lastSlot *int
}

func (a *toolCallAccumulator) feed(deltas []toolCallDelta) {
	for i := range deltas {
		a.merge(&deltas[i])
	}
}

func (a *toolCallAccumulator) slotFor(delta *toolCallDelta) int {
	if delta.Index != nil {
		return *delta.Index
	}
	// Providers that omit `index` start a new call whenever they send a fresh
	// `id`; everything else continues the call already in progress.
	if delta.ID != "" || a.lastSlot == nil {
		next := 0
		for slot := range a.slots {
			if slot >= next {
				next = slot + 1
			}
		}
		return next
	}
	return *a.lastSlot
}

func (a *toolCallAccumulator) merge(delta *toolCallDelta) {
	if a.slots == nil {
		a.slots = make(map[int]*ToolCall)
	}
	slot := a.slotFor(delta)
	a.lastSlot = &slot

	call, ok := a.slots[slot]
	if !ok {
		call = &ToolCall{ID: "", Type: "", Function: ToolCallFunction{Name: "", Arguments: ""}, Extra: nil}
		a.slots[slot] = call
	}

	// Later frames repeat these as empty strings; keep the first real one.
	if delta.ID != "" {
		call.ID = delta.ID
	}
	if delta.Type != "" {
		call.Type = delta.Type
	}
	for key, value := range delta.Extra {
		if call.Extra == nil {
			call.Extra = make(map[string]json.RawMessage, len(delta.Extra))
		}
		call.Extra[key] = value
	}
	if delta.Function == nil {
		return
	}
	if delta.Function.Name != "" {
		call.Function.Name = delta.Function.Name
	}
	if delta.Function.Arguments != nil {
		call.Function.Arguments += *delta.Function.Arguments
	}
}

// result returns the assembled calls, ordered by their provider-assigned index.
func (a *toolCallAccumulator) result() []ToolCall {
	if len(a.slots) == 0 {
		return nil
	}
	slots := make([]int, 0, len(a.slots))
	for slot := range a.slots {
		slots = append(slots, slot)
	}
	sort.Ints(slots)

	calls := make([]ToolCall, 0, len(slots))
	for _, slot := range slots {
		calls = append(calls, *a.slots[slot])
	}
	return calls
}
