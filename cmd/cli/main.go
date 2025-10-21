package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/rocry/smolllm-go/smolllm"
)

type cli struct {
	Model          string        `help:"Provider/model identifier (e.g. openai/gpt-4). Falls back to SMOLLLM_MODEL." short:"m"`
	System         string        `help:"Optional system prompt." short:"s"`
	Images         []string      `help:"Optional image paths or data URLs."`
	Timeout        time.Duration `help:"Overall timeout for the request." default:"120s"`
	StripBackticks bool          `help:"Remove enclosing markdown backticks before printing."`
	Stream         bool          `help:"Stream tokens to stdout as they arrive."`
	Prompt         []string      `arg:"" name:"prompt" help:"Prompt text to send." type:"string"`
}

func (c *cli) Run() error {
	if len(c.Prompt) == 0 {
		return errors.New("prompt text is required")
	}

	prompt := strings.Join(c.Prompt, " ")
	options := []smolllm.Option{
		smolllm.WithTimeout(c.Timeout),
	}

	if c.Model != "" {
		options = append(options, smolllm.WithModel(c.Model))
	}
	if c.System != "" {
		options = append(options, smolllm.WithSystemPrompt(c.System))
	}
	if len(c.Images) > 0 {
		options = append(options, smolllm.WithImagePaths(c.Images...))
	}

	ctx := context.Background()

	if c.Stream {
		streamResp, err := smolllm.Stream(ctx, smolllm.PromptFromString(prompt), options...)
		if err != nil {
			return err
		}
		for delta := range streamResp.Stream.Chan() {
			_, _ = fmt.Fprint(os.Stdout, delta)
		}
		if err := streamResp.Stream.Wait(); err != nil {
			return err
		}
		_, _ = fmt.Fprintln(os.Stdout)
		return nil
	}

	if c.StripBackticks {
		options = append(options, smolllm.WithBacktickRemoval())
	}

	resp, err := smolllm.Ask(ctx, smolllm.PromptFromString(prompt), options...)
	if err != nil {
		return err
	}
	fmt.Println(resp.Text)
	return nil
}

func main() {
	cliCfg := &cli{}
	parser := kong.Parse(cliCfg,
		kong.Name("smolllm-go"),
		kong.Description("Minimal LLM CLI compatible with OpenAI-style chat completions."),
	)
	if err := parser.Run(); err != nil {
		log.Fatalf("error: %v", err)
	}
}
