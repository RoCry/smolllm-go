package smolllm

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared/constant"
)

// Message is compatible with the OpenAI Go SDK chat completion message union.
type Message = openai.ChatCompletionMessageParamUnion

// Prompt is the request payload passed to Ask/Stream.
type Prompt struct {
	Messages []Message
}

// PromptFromString creates a single user message prompt.
func PromptFromString(text string) Prompt {
	msg := openai.UserMessage(text)
	ensureRole(&msg)
	return Prompt{
		Messages: []Message{msg},
	}
}

// PromptFromMessages constructs a prompt from an existing slice.
func PromptFromMessages(messages []Message) Prompt {
	cp := make([]Message, len(messages))
	copy(cp, messages)
	for i := range cp {
		ensureRole(&cp[i])
	}
	return Prompt{Messages: cp}
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

// Validate ensures the prompt is well formed.
func (p Prompt) Validate() error {
	if len(p.Messages) == 0 {
		return errors.New("prompt must contain at least one message")
	}
	for i, msg := range p.Messages {
		if _, ok := messageRole(msg); !ok {
			return fmt.Errorf("prompt message #%d must set role", i)
		}

		if role, _ := messageRole(msg); role == "tool" || role == "function" {
			return fmt.Errorf("prompt message #%d uses unsupported role %q", i, role)
		}

		if content := msg.GetContent().AsAny(); content != nil {
			continue
		}

		if toolCalls := msg.GetToolCalls(); len(toolCalls) > 0 {
			continue
		}

		if msg.GetFunctionCall() != nil {
			continue
		}

		return fmt.Errorf("prompt message #%d must set content", i)
	}
	return nil
}

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
	}
}

// StreamChunk carries a single streamed delta with optional reasoning.
type StreamChunk struct {
	Content   string
	Reasoning string
}

// String returns only the content portion (backward-compatible with fmt.Fprint).
func (c StreamChunk) String() string { return c.Content }

// IsEmpty reports whether the chunk carries no content and no reasoning.
func (c StreamChunk) IsEmpty() bool { return c.Content == "" && c.Reasoning == "" }

// Usage captures per-call metrics and routing details.
type Usage struct {
	Provider     string        `json:"provider"`
	Model        string        `json:"model"`
	ModelName    string        `json:"model_name"`
	APIKeyHint   string        `json:"api_key_hint"`
	InputTokens  int           `json:"input_tokens"`
	OutputTokens int           `json:"output_tokens"`
	Duration     time.Duration `json:"duration"`
	TTFT         time.Duration `json:"ttft"`
}

// RequestEvent is emitted after each LLM call attempt (success or failure).
type RequestEvent struct {
	Usage
	Error     error     `json:"-"`
	Timestamp time.Time `json:"timestamp"`
}

// Response holds a full LLM response.
type Response struct {
	Text      string `json:"text"`
	Reasoning string `json:"reasoning"`
	Model     string `json:"model"`
	ModelName string `json:"model_name"`
	Provider  string `json:"provider"`
	Usage     Usage  `json:"usage"`
}

// DeltaStream represents a streaming LLM response.
type DeltaStream struct {
	ch        <-chan StreamChunk
	done      <-chan streamCompletion
	cancel    func()
	logger    *slog.Logger
	reasoning *string // populated by Wait() from streamCompletion
	usage     *Usage
	hook      func(RequestEvent)
}

type streamCompletion struct {
	err       error
	metrics   *streamMetrics
	reasoning string
}

type streamMetrics struct {
	modelName    string
	inputTokens  int
	outputTokens int
	total        time.Duration
	ttft         time.Duration
}

// Chan exposes the underlying channel of chunks.
func (s DeltaStream) Chan() <-chan StreamChunk {
	return s.ch
}

// Close cancels the stream.
func (s DeltaStream) Close() {
	if s.cancel != nil {
		s.cancel()
	}
}

// Wait blocks until the stream finishes and returns the terminal error.
// After Wait returns, the parent StreamResponse.Reasoning and Usage are populated.
func (s DeltaStream) Wait() error {
	if s.done == nil {
		return nil
	}
	result := <-s.done
	if s.reasoning != nil {
		*s.reasoning = result.reasoning
	}
	if result.metrics != nil {
		if s.logger != nil {
			s.logger.Info(
				formatMetrics(
					result.metrics.modelName,
					result.metrics.inputTokens,
					result.metrics.outputTokens,
					result.metrics.total,
					result.metrics.ttft,
				),
				"model", result.metrics.modelName,
			)
		}
		if s.usage != nil {
			s.usage.InputTokens = result.metrics.inputTokens
			s.usage.OutputTokens = result.metrics.outputTokens
			s.usage.Duration = result.metrics.total
			s.usage.TTFT = result.metrics.ttft
		}
	}
	if s.hook != nil && s.usage != nil {
		s.hook(RequestEvent{
			Usage:     *s.usage,
			Error:     result.err,
			Timestamp: time.Now().UTC(),
		})
	}
	return result.err
}

// StreamResponse wraps streaming metadata.
type StreamResponse struct {
	Stream    DeltaStream `json:"-"`
	Reasoning string      `json:"reasoning"`
	Model     string      `json:"model"`
	ModelName string      `json:"model_name"`
	Provider  string      `json:"provider"`
	Usage     Usage       `json:"usage"`
}
