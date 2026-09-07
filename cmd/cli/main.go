// Package main implements the smolllm CLI.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"math"
	"os"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/rocry/smolllm-go/smolllm"
)

type rootCmd struct {
	Ask   askCmd   `cmd:"" default:"withargs" help:"Chat completion (default)."`
	Embed embedCmd `cmd:"" help:"Generate embeddings."`
}

// askCmd exposes a tiny surface: API keys come from {PROVIDER}_API_KEY env vars and
// support comma-separated values for automatic rotation, while models accept the
// same comma pattern for ordered fallbacks.
type askCmd struct {
	Model           string        `help:"Provider/model. Env SMOLLLM_MODEL. Comma fallbacks." short:"m"`
	System          string        `help:"System prompt injected as the first message." short:"s"`
	Images          []string      `help:"Image paths or data URLs for multimodal prompts."`
	Temperature     *float64      `help:"Sampling temperature in [0,2]."`
	TopP            *float64      `help:"Nucleus sampling cutoff probability in [0,1]." name:"top-p"`
	ReasoningEffort *string       `help:"Reasoning effort (e.g. none/low/medium/high)." name:"reasoning-effort"`
	Timeout         time.Duration `help:"Overall timeout for the request." default:"120s"`
	StripBackticks  bool          `help:"Remove enclosing markdown backticks before printing."`
	Stream          bool          `help:"Stream tokens to stdout as they arrive."`
	Validate        bool          `help:"Validate API configuration and exit without sending prompt."`
	Prompt          []string      `arg:"" name:"prompt" help:"Prompt text to send." type:"string" optional:""`
}

type embedCmd struct {
	Model           string        `help:"Provider/model id (e.g. ollama/qwen3-embedding:0.6b)." short:"m"`
	Timeout         time.Duration `help:"Overall timeout for the request." default:"60s"`
	Format          string        `help:"Output format: json or tsv." default:"json" enum:"json,tsv"`
	Dimensions      int           `help:"Truncate output vectors to N dimensions (requires MRL-capable model)." short:"d"`
	ReasoningEffort *string       `help:"Reasoning effort (e.g. none) to disable thinking." name:"reasoning-effort"`
	Inputs          []string      `arg:"" name:"input" help:"Text inputs to embed (one embedding per arg)." required:""`
}

func (c *askCmd) options() ([]smolllm.Option, error) {
	options := []smolllm.Option{
		smolllm.WithLogger(cliLogger()),
		smolllm.WithTimeout(c.Timeout),
	}

	if trimmed := strings.TrimSpace(c.Model); trimmed != "" {
		options = append(options, smolllm.WithModel(trimmed))
	}
	if c.Temperature != nil {
		if math.IsNaN(*c.Temperature) {
			return nil, errors.New("temperature cannot be NaN")
		}
		if *c.Temperature < 0 || *c.Temperature > 2 {
			return nil, errors.New("temperature must be between 0 and 2 inclusive")
		}
		options = append(options, smolllm.WithTemperature(*c.Temperature))
	}
	if c.TopP != nil {
		if math.IsNaN(*c.TopP) {
			return nil, errors.New("top-p cannot be NaN")
		}
		if *c.TopP < 0 || *c.TopP > 1 {
			return nil, errors.New("top-p must be between 0 and 1 inclusive")
		}
		options = append(options, smolllm.WithTopP(*c.TopP))
	}
	if c.ReasoningEffort != nil {
		options = append(options, smolllm.WithReasoningEffort(*c.ReasoningEffort))
	}
	if len(c.Images) > 0 {
		options = append(options, smolllm.WithImagePaths(c.Images...))
	}
	if c.StripBackticks {
		options = append(options, smolllm.WithBacktickRemoval())
	}
	return options, nil
}

func (c *askCmd) Run() error {
	options, err := c.options()
	if err != nil {
		return err
	}

	client := smolllm.New(options...)
	if err := client.Validate(); err != nil {
		return err
	}
	if c.Validate {
		fmt.Println("✓ API configuration is valid")
		return nil
	}

	promptText := strings.TrimSpace(strings.Join(c.Prompt, " "))
	if promptText == "" {
		return errors.New("prompt text is required")
	}

	req := smolllm.RequestFromString(promptText)
	req.System = strings.TrimSpace(c.System)

	ctx := context.Background()
	if c.Stream {
		// The deltas were printed as they arrived, so only the outcome is left.
		return reportOutcome(streamToStdout(client.Stream(ctx, req)))
	}
	return reportResult(client.Ask(ctx, req))
}

// streamToStdout prints deltas as they arrive: answer text on stdout, thinking
// on stderr, so a piped caller gets only the answer.
func streamToStdout(stream *smolllm.EventStream) *smolllm.AssistantMessage {
	inReasoning := false
	for event := range stream.Events() {
		switch event.Kind {
		case smolllm.EventReasoningDelta:
			if !inReasoning {
				_, _ = fmt.Fprintf(os.Stderr, "[Thinking]\n")
				inReasoning = true
			}
			_, _ = fmt.Fprint(os.Stderr, event.Delta)
		case smolllm.EventTextDelta:
			if inReasoning {
				_, _ = fmt.Fprintf(os.Stderr, "\n[Answer]\n")
				inReasoning = false
			}
			_, _ = fmt.Fprint(os.Stdout, event.Delta)
		case smolllm.EventToolCallEnd:
			if event.ToolCall != nil {
				_, _ = fmt.Fprintf(os.Stderr, "\n[Tool call] %s %s\n",
					event.ToolCall.Function.Name, event.ToolCall.Function.Arguments)
			}
		case smolllm.EventLegFailed:
			if event.Attempt != nil && event.Attempt.Err != nil {
				_, _ = fmt.Fprintf(os.Stderr, "\n[leg failed] %v\n", event.Attempt.Err)
			}
		case smolllm.EventStart, smolllm.EventToolCallStart, smolllm.EventToolCallDelta,
			smolllm.EventDone, smolllm.EventError:
			// Nothing to print: the terminal message carries the outcome.
		}
	}
	_, _ = fmt.Fprintln(os.Stdout)
	return stream.Result()
}

// reportOutcome turns a never-throwing call into a process exit status, without
// printing the answer: a streaming caller has already seen it.
func reportOutcome(msg *smolllm.AssistantMessage) error {
	switch msg.StopReason {
	case smolllm.StopReasonError:
		return errors.New(msg.ErrorMessage)
	case smolllm.StopReasonAborted:
		return fmt.Errorf("aborted: %s", msg.ErrorMessage)
	case smolllm.StopReasonPending, smolllm.StopReasonStop,
		smolllm.StopReasonLength, smolllm.StopReasonToolUse:
	}
	if msg.StopReason == smolllm.StopReasonLength {
		_, _ = fmt.Fprintln(os.Stderr, "[truncated] the model hit its output limit")
	}
	return nil
}

// reportResult prints a completed answer and turns the call into an exit status.
func reportResult(msg *smolllm.AssistantMessage) error {
	if err := reportOutcome(msg); err != nil {
		return err
	}
	if msg.Reasoning != "" {
		_, _ = fmt.Fprintf(os.Stderr, "[reasoning] %s\n", msg.Reasoning)
	}
	if msg.Content != "" {
		fmt.Println(msg.Content)
	}
	for _, call := range msg.ToolCalls {
		_, _ = fmt.Fprintf(os.Stderr, "[tool call] %s %s\n", call.Function.Name, call.Function.Arguments)
	}
	return nil
}

func (c *embedCmd) Run() error {
	options := []smolllm.Option{
		smolllm.WithLogger(cliLogger()),
		smolllm.WithTimeout(c.Timeout),
	}

	if trimmed := strings.TrimSpace(c.Model); trimmed != "" {
		options = append(options, smolllm.WithModel(trimmed))
	}
	if c.Dimensions > 0 {
		options = append(options, smolllm.WithDimensions(c.Dimensions))
	} else if c.Dimensions < 0 {
		return errors.New("dimensions must be positive")
	}
	if c.ReasoningEffort != nil {
		options = append(options, smolllm.WithReasoningEffort(*c.ReasoningEffort))
	}

	resp, err := smolllm.New(options...).Embed(context.Background(), c.Inputs)
	if err != nil {
		return err
	}

	switch c.Format {
	case "tsv":
		for _, vec := range resp.Embeddings {
			parts := make([]string, len(vec))
			for i, v := range vec {
				parts[i] = fmt.Sprintf("%g", v)
			}
			fmt.Println(strings.Join(parts, "\t"))
		}
	default:
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(resp)
	}
	return nil
}

func main() {
	var root rootCmd
	parser := kong.Parse(&root,
		kong.Name("smolllm"),
		kong.Description("Minimal LLM CLI compatible with OpenAI-style APIs."),
	)
	if err := parser.Run(); err != nil {
		log.Fatalf("error: %v", err)
	}
}

func cliLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		AddSource: false,
		Level:     slog.LevelInfo,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.TimeKey {
				return slog.Attr{
					Key:   attr.Key,
					Value: slog.StringValue(attr.Value.Time().UTC().Format(time.RFC3339)),
				}
			}
			return attr
		},
	}))
}
