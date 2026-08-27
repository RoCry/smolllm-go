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

## Model Format

- `provider/model[!effort]` — provider resolves base URL and API key from the table/env by name
- `model[!effort]` (bare, no `/`) — no provider: base URL and API key must come from `WithBaseURL`/`WithAPIKey` explicitly (no env fallback); provider surfaces as `""` in usage, hooks, and logs
- Comma-separated specs form a fallback chain and may mix both forms
- An explicit `WithBaseURL` overrides ALL legs of a chain (existing precedence: explicit > env > provider table)

## CLI

```
go run ./cmd/smolllm-go --model gemini/gemini-2.0-flash "Say hello world"
```

Reasoning effort can be passed globally or as a per-model fallback suffix:

```
go run ./cmd/smolllm-go --model 'groq/qwen/qwen3-32b!none,gemini/gemini-3.1-flash-lite-preview' "Say hello"
go run ./cmd/smolllm-go --model openai/gpt-5 --reasoning-effort medium "Say hello"
```

Flags:
- `--stream` stream deltas instead of waiting for completion
- `--system` inject system message
- `--image` attach image path or data URL (repeatable)
- `--temperature` control sampling randomness `[0,2]`
- `--top-p` control nucleus sampling cutoff `[0,1]`
- `--reasoning-effort` pass `none|minimal|low|medium|high|xhigh` to compatible providers
- `--timeout` override default `600s`
- `--strip-backticks` remove enclosing markdown fences

## Env Layout

- `SMOLLLM_MODEL` fallback when no `WithModel`
- `${PROVIDER}_API_KEY` comma list allowed
- `${PROVIDER}_BASE_URL` optional override, matches provider slug (hyphen → underscore)
- Bare models (no `provider/` prefix) never read env — explicit options only
- `LOG_LEVEL` optional (`DEBUG`, `INFO`, `WARN`, `ERROR`)

## Features

- key/base-url load balancing with usage tracking
- streaming via `smolllm.Stream` returning `DeltaStream`
- image prompts via `WithImagePaths`
- markdown fence stripping via `WithBacktickRemoval`
- fail-fast validation for env, prompts, and responses, including proactive model/API-key checks via `smolllm.Validate` (invoked automatically by the CLI)
- raw request fields the library does not model via `WithExtraBody`

## Escape Hatch

`WithExtraBody` merges raw request fields into the payload last, so they win over
library defaults:

```go
resp, err := smolllm.Ask(ctx, prompt,
    smolllm.WithModel("openai/gpt-4o-mini"),
    smolllm.WithExtraBody(map[string]any{"response_format": map[string]any{"type": "json_object"}}),
)
```

The fields the library reads back — `stream`, `stream_options`, `messages`,
`model` — are rejected.

## Tests

```
go test ./...
```
