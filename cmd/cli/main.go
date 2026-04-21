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
	ReasoningEffort *string       `help:"Reasoning effort (e.g. none) to disable thinking." name:"reasoning-effort"`
	Inputs          []string      `arg:"" name:"input" help:"Text inputs to embed (one embedding per arg)." required:""`
}

func (c *askCmd) Run() error {
	options := []smolllm.Option{
		smolllm.WithLogger(cliLogger()),
	}

	if trimmed := strings.TrimSpace(c.System); trimmed != "" {
		options = append(options, smolllm.WithSystemPrompt(trimmed))
	}
	if trimmed := strings.TrimSpace(c.Model); trimmed != "" {
		options = append(options, smolllm.WithModel(trimmed))
	}
	if c.Temperature != nil {
		if math.IsNaN(*c.Temperature) {
			return fmt.Errorf("temperature cannot be NaN")
		}
		if *c.Temperature < 0 || *c.Temperature > 2 {
			return fmt.Errorf("temperature must be between 0 and 2 inclusive")
		}
		options = append(options, smolllm.WithTemperature(*c.Temperature))
	}
	if c.TopP != nil {
		if math.IsNaN(*c.TopP) {
			return fmt.Errorf("top-p cannot be NaN")
		}
		if *c.TopP < 0 || *c.TopP > 1 {
			return fmt.Errorf("top-p must be between 0 and 1 inclusive")
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

	if err := smolllm.Validate(options...); err != nil {
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

	prompt := smolllm.PromptFromString(promptText)

	ctx, cancel := deriveCLIContext(context.Background(), c.Timeout)
	defer cancel()

	if c.Stream {
		resp, err := smolllm.Stream(ctx, prompt, options...)
		if err != nil {
			return err
		}
		inReasoning := false
		for chunk := range resp.Stream.Chan() {
			if chunk.Reasoning != "" {
				if !inReasoning {
					_, _ = fmt.Fprintf(os.Stderr, "[Thinking]\n")
					inReasoning = true
				}
				_, _ = fmt.Fprint(os.Stderr, chunk.Reasoning)
			}
			if chunk.Content != "" {
				if inReasoning {
					_, _ = fmt.Fprintf(os.Stderr, "\n[Answer]\n")
					inReasoning = false
				}
				_, _ = fmt.Fprint(os.Stdout, chunk.Content)
			}
		}
		if err := resp.Stream.Wait(); err != nil {
			return err
		}
		_, _ = fmt.Fprintln(os.Stdout)
		return nil
	}

	resp, err := smolllm.Ask(ctx, prompt, options...)
	if err != nil {
		return err
	}
	if resp.Reasoning != "" {
		_, _ = fmt.Fprintf(os.Stderr, "[reasoning] %s\n", resp.Reasoning)
	}
	fmt.Println(resp.Text)
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
	if c.ReasoningEffort != nil {
		options = append(options, smolllm.WithReasoningEffort(*c.ReasoningEffort))
	}

	ctx, cancel := deriveCLIContext(context.Background(), c.Timeout)
	defer cancel()

	resp, err := smolllm.Embed(ctx, c.Inputs, options...)
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

func deriveCLIContext(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return context.WithCancel(parent)
	}
	return context.WithTimeout(parent, timeout)
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
