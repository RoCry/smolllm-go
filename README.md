# SmolLLM Go

Minimal Go client for OpenAI-compatible chat completions with multi-provider routing.

## Installation

```
go get github.com/rocry/smolllm-go/smolllm
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"

    "github.com/rocry/smolllm-go/smolllm"
)

func main() {
    resp, err := smolllm.Ask(
        context.Background(),
        smolllm.PromptFromString("Say hello world"),
        smolllm.WithModel("gemini/gemini-2.0-flash"),
    )
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Text)
}
```

## CLI

```
go run ./cmd/smolllm-go --model gemini/gemini-2.0-flash "Say hello world"
```

Flags:
- `--stream` stream deltas instead of waiting for completion
- `--system` inject system message
- `--image` attach image path or data URL (repeatable)
- `--timeout` override default `120s`
- `--strip-backticks` remove enclosing markdown fences

## Env Layout

- `SMOLLLM_MODEL` fallback when no `WithModel`
- `${PROVIDER}_API_KEY` comma list allowed
- `${PROVIDER}_BASE_URL` optional override, matches provider slug (hyphen → underscore)
- `LOG_LEVEL` optional (`DEBUG`, `INFO`, `WARN`, `ERROR`)

## Features

- key/base-url load balancing with usage tracking
- streaming via `smolllm.Stream` returning `DeltaStream`
- image prompts via `WithImagePaths`
- markdown fence stripping via `WithBacktickRemoval`
- fail-fast validation for env, prompts, and responses, including proactive model/API-key checks via `smolllm.Validate` (invoked automatically by the CLI)

## Tests

```
go test ./...
```
