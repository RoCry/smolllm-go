package smolllm

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared/constant"
)

// Message is compatible with the OpenAI Go SDK chat completion message union.
type Message = openai.ChatCompletionMessageParamUnion

// Request is one conversational turn: the messages, plus the system prompt and
// tools in force for it. Routing and sampling live in Options instead.
type Request struct {
	// System is prepended as a system message when non-empty.
	System string
	// Messages is the conversation so far.
	Messages []Message
	// Tools declares the functions the model may call. Omitted from the wire
	// when empty.
	Tools []Tool
}

// Tool declares a function the model may call. Parameters is a raw JSON Schema
// object, passed through untouched: smolllm never inspects or repairs it.
type Tool struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// RequestFromString builds a single-user-message Request.
func RequestFromString(text string) Request {
	msg := openai.UserMessage(text)
	ensureRole(&msg)
	return Request{
		System:   "",
		Messages: []Message{msg},
		Tools:    nil,
	}
}

// RequestFromMessages copies messages into a Request.
func RequestFromMessages(messages []Message) Request {
	cp := make([]Message, len(messages))
	copy(cp, messages)
	for i := range cp {
		ensureRole(&cp[i])
	}
	return Request{System: "", Messages: cp, Tools: nil}
}

// System returns a system role chat message.
func System(content string) Message {
	msg := openai.SystemMessage(content)
	ensureRole(&msg)
	return msg
}

// User returns a user role chat message.
func User(content string) Message {
	msg := openai.UserMessage(content)
	ensureRole(&msg)
	return msg
}

// Assistant returns an assistant role chat message.
func Assistant(content string) Message {
	msg := openai.AssistantMessage(content)
	ensureRole(&msg)
	return msg
}

// Developer returns a developer role chat message.
func Developer(content string) Message {
	msg := openai.DeveloperMessage(content)
	ensureRole(&msg)
	return msg
}

// AssistantToolCalls returns an assistant message replaying the tool calls the
// model asked for. Pass empty text when the turn carried no content.
func AssistantToolCalls(text string, calls []ToolCall) Message {
	params := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(calls))
	for _, call := range calls {
		fn := openai.ChatCompletionMessageFunctionToolCallParam{ //nolint:exhaustruct_v5 // Type defaults to "function"
			ID: call.ID,
			Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
				Name:      call.Function.Name,
				Arguments: call.Function.Arguments,
			},
		}
		// Provider extras (e.g. Gemini thought signatures) must survive replay.
		if extra := call.extraFields(); extra != nil {
			fn.SetExtraFields(extra)
		}
		params = append(params, openai.ChatCompletionMessageToolCallUnionParam{ //nolint:exhaustruct_v5 // union arm
			OfFunction: &fn,
		})
	}
	assistant := openai.ChatCompletionAssistantMessageParam{ //nolint:exhaustruct_v5 // optional fields stay unset
		ToolCalls: params,
	}
	if text != "" {
		assistant.Content = openai.ChatCompletionAssistantMessageParamContentUnion{ //nolint:exhaustruct_v5 // text arm
			OfString: openai.String(text),
		}
	}
	msg := Message{OfAssistant: &assistant} //nolint:exhaustruct_v5 // union arm
	ensureRole(&msg)
	return msg
}

// AttachReasoning replays reasoning on an assistant message as the
// reasoning_content field, empty included: DeepSeek thinking mode rejects a
// replayed tool-call turn it did not issue unless the field is present.
// Attaching is the caller's decision, because a provider that rejects unknown
// message fields must never receive it.
func AttachReasoning(msg Message, reasoning string) Message {
	if msg.OfAssistant == nil {
		panic("AttachReasoning: message is not an assistant message")
	}
	assistant := *msg.OfAssistant
	assistant.SetExtraFields(map[string]any{"reasoning_content": reasoning})
	msg.OfAssistant = &assistant
	return msg
}

// ToolResult returns a tool role message carrying the output of one tool call.
func ToolResult(toolCallID, content string) Message {
	msg := openai.ToolMessage(content, toolCallID)
	ensureRole(&msg)
	return msg
}

// Validate reports a malformed Request: no messages, or a message missing its
// role or content.
func (r Request) Validate() error {
	if len(r.Messages) == 0 {
		return errors.New("request must contain at least one message")
	}
	for i, msg := range r.Messages {
		if _, ok := messageRole(msg); !ok {
			return fmt.Errorf("request message #%d must set role", i)
		}

		// The legacy `function` role is deprecated upstream and stays rejected;
		// `tool` is how a caller replays a tool result.
		if role, _ := messageRole(msg); role == roleFunction {
			return fmt.Errorf("request message #%d uses unsupported role %q", i, role)
		}

		if content := msg.GetContent().AsAny(); content != nil {
			continue
		}

		if msg.OfTool != nil {
			// A tool result may legitimately be an empty string.
			continue
		}

		if toolCalls := msg.GetToolCalls(); len(toolCalls) > 0 {
			continue
		}

		if msg.GetFunctionCall() != nil {
			continue
		}

		return fmt.Errorf("request message #%d must set content", i)
	}
	return nil
}

// roleFunction is the deprecated OpenAI `function` message role. It is
// recognized only so a request carrying it is rejected with a clear error.
const roleFunction = "function"

func messageRole(msg Message) (string, bool) {
	if role := msg.GetRole(); role != nil {
		if trimmed := strings.TrimSpace(*role); trimmed != "" {
			return trimmed, true
		}
	}
	switch {
	case msg.OfDeveloper != nil:
		return "developer", true
	case msg.OfSystem != nil:
		return "system", true
	case msg.OfUser != nil:
		return "user", true
	case msg.OfAssistant != nil:
		return "assistant", true
	case msg.OfTool != nil:
		return "tool", true
	case msg.OfFunction != nil:
		return roleFunction, true
	default:
		return "", false
	}
}

func ensureRole(msg *Message) {
	if msg == nil {
		return
	}
	switch {
	case msg.OfDeveloper != nil:
		msg.OfDeveloper.Role = constant.ValueOf[constant.Developer]()
	case msg.OfSystem != nil:
		msg.OfSystem.Role = constant.ValueOf[constant.System]()
	case msg.OfUser != nil:
		msg.OfUser.Role = constant.ValueOf[constant.User]()
	case msg.OfAssistant != nil:
		msg.OfAssistant.Role = constant.ValueOf[constant.Assistant]()
	case msg.OfTool != nil:
		msg.OfTool.Role = constant.ValueOf[constant.Tool]()
	}
}
