package smolllm

import (
	"errors"
	"strconv"
)

// Role is a chat completion message role.
type Role string

const (
	// RoleUser represents end-user authored content.
	RoleUser Role = "user"
	// RoleAssistant represents model responses.
	RoleAssistant Role = "assistant"
	// RoleSystem represents system instructions.
	RoleSystem Role = "system"
)

// Message is a single chat completion message.
type Message struct {
	Role    Role        `json:"role"`
	Content interface{} `json:"content"`
}

// Prompt is the request payload passed to Ask/Stream.
type Prompt struct {
	Messages []Message
}

// PromptFromString creates a single user message prompt.
func PromptFromString(text string) Prompt {
	return Prompt{
		Messages: []Message{
			{Role: RoleUser, Content: text},
		},
	}
}

// PromptFromMessages constructs a prompt from an existing slice.
func PromptFromMessages(messages []Message) Prompt {
	cp := make([]Message, len(messages))
	copy(cp, messages)
	return Prompt{Messages: cp}
}

// Validate ensures the prompt is well formed.
func (p Prompt) Validate() error {
	if len(p.Messages) == 0 {
		return errors.New("prompt must contain at least one message")
	}
	for i, msg := range p.Messages {
		if msg.Role == "" {
			return errors.New("prompt message #" + strconv.Itoa(i) + " must set role")
		}
		if msg.Content == nil {
			return errors.New("prompt message #" + strconv.Itoa(i) + " must set content")
		}
	}
	return nil
}

// Response holds a full LLM response.
type Response struct {
	Text      string `json:"text"`
	Model     string `json:"model"`
	ModelName string `json:"model_name"`
	Provider  string `json:"provider"`
}

// DeltaStream represents a streaming LLM response.
type DeltaStream struct {
	ch     <-chan string
	done   <-chan error
	cancel func()
}

// Chan exposes the underlying channel of chunks.
func (s DeltaStream) Chan() <-chan string {
	return s.ch
}

// Close cancels the stream.
func (s DeltaStream) Close() {
	if s.cancel != nil {
		s.cancel()
	}
}

// Wait blocks until the stream finishes and returns the terminal error.
func (s DeltaStream) Wait() error {
	if s.done == nil {
		return nil
	}
	return <-s.done
}

// Done exposes the completion channel.
func (s DeltaStream) Done() <-chan error {
	return s.done
}

// StreamResponse wraps streaming metadata.
type StreamResponse struct {
	Stream    DeltaStream `json:"-"`
	Model     string      `json:"model"`
	ModelName string      `json:"model_name"`
	Provider  string      `json:"provider"`
}
