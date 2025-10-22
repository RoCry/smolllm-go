package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/rocry/smolllm-go/smolllm"
)

// cli exposes a tiny surface: API keys come from {PROVIDER}_API_KEY env vars and
// support comma-separated values for automatic rotation, while models accept the
// same comma pattern for ordered fallbacks.
type cli struct {
	Model          string        `help:"Provider/model id (openai/gpt-4o-mini). SMOLLLM_MODEL default. Comma fallbacks." short:"m"`
	System         string        `help:"Optional system prompt injected as the first message." short:"s"`
	Images         []string      `help:"Optional image paths or data URLs for multimodal prompts."`
	Timeout        time.Duration `help:"Overall timeout for the request." default:"120s"`
	StripBackticks bool          `help:"Remove enclosing markdown backticks before printing."`
	Stream         bool          `help:"Stream tokens to stdout as they arrive."`
	Validate       bool          `help:"Validate API configuration and exit without sending prompt."`
	Prompt         []string      `arg:"" name:"prompt" help:"Prompt text to send." type:"string" optional:""`
}

func (c *cli) Run() error {
	options := []smolllm.Option{
		smolllm.WithLogger(cliLogger()),
	}

	if trimmed := strings.TrimSpace(c.System); trimmed != "" {
		options = append(options, smolllm.WithSystemPrompt(trimmed))
	}
	if trimmed := strings.TrimSpace(c.Model); trimmed != "" {
		// Comma separated model list gives ordered fallbacks handled by smolllm.
		options = append(options, smolllm.WithModel(trimmed))
	}
	if len(c.Images) > 0 {
		// Multiple images accepted; smolllm converts each to the OpenAI image_url shape.
		options = append(options, smolllm.WithImagePaths(c.Images...))
	}
	if c.StripBackticks {
		options = append(options, smolllm.WithBacktickRemoval())
	}

	// Validate-only mode: check configuration and exit
	if c.Validate {
		if err := smolllm.Validate(options...); err != nil {
			return err
		}
		fmt.Println("✓ API configuration is valid")
		return nil
	}

	// Normal mode: require prompt and validate before execution
	promptText := strings.TrimSpace(strings.Join(c.Prompt, " "))
	if promptText == "" {
		return errors.New("prompt text is required")
	}

	if err := smolllm.Validate(options...); err != nil {
		return err
	}

	prompt := c.buildPrompt(promptText)

	ctx, cancel := deriveCLIContext(context.Background(), c.Timeout)
	defer cancel()

	if c.Stream {
		resp, err := smolllm.Stream(ctx, prompt, options...)
		if err != nil {
			return err
		}
		for delta := range resp.Stream.Chan() {
			_, _ = fmt.Fprint(os.Stdout, delta)
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
	fmt.Println(resp.Text)
	return nil
}

func deriveCLIContext(parent context.Context, timeout time.Duration) (context.Context, context.CancelFunc) {
	if timeout <= 0 {
		return context.WithCancel(parent)
	}
	return context.WithTimeout(parent, timeout)
}

func (c *cli) buildPrompt(userText string) smolllm.Prompt {
	return smolllm.PromptFromString(userText)
}

func main() {
	cliCfg := new(cli)
	parser := kong.Parse(cliCfg,
		kong.Name("smolllm"),
		kong.Description("Minimal LLM CLI compatible with OpenAI-style chat completions."),
	)
	if err := parser.Run(); err != nil {
		log.Fatalf("error: %v", err)
	}
}

func cliLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		AddSource: false,
		Level:     slog.LevelInfo,
		ReplaceAttr: func(groups []string, attr slog.Attr) slog.Attr {
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
