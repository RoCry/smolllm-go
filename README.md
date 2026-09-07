# SmolLLM Go

Minimal Go client for OpenAI-compatible chat completions with multi-provider routing.

## Installation

```
go get github.com/rocry/smolllm-go/smolllm
```

## Quick Start

A call never reports an operational failure as a Go error. `Ask` always returns a
message; `StopReason` says how the call ended.

```go
package main

import (
    "context"
    "fmt"

    "github.com/rocry/smolllm-go/smolllm"
)

func main() {
    client := smolllm.New(smolllm.WithModel("gemini/gemini-2.0-flash"))

    msg := client.Ask(context.Background(), smolllm.RequestFromString("Say hello world"))
    if msg.StopReason == smolllm.StopReasonError {
        panic(msg.ErrorMessage)
    }
    fmt.Println(msg.Content)
}
```

## Streaming

`Stream` returns immediately. `Events()` and `Result()` are independent: take
either, or both.

```go
stream := client.Stream(ctx, smolllm.RequestFromString("Write a haiku"))

for event := range stream.Events() {
    switch event.Kind {
    case smolllm.EventTextDelta:
        fmt.Print(event.Delta)
    case smolllm.EventReasoningDelta:
        fmt.Fprint(os.Stderr, event.Delta)
    case smolllm.EventToolCallDelta:
        // Raw argument JSON, as it streams.
    case smolllm.EventToolCallEnd:
        dispatch(event.ToolCall)
    case smolllm.EventLegFailed:
        log.Printf("leg failed: %v", event.Attempt.Err)
    }
}

msg := stream.Result() // never nil, never an error
```

Every `Event` carries `Message`, an immutable snapshot of the turn so far. A
caller that takes `Result()` without draining `Events()` should `Close()` the
stream to release the goroutine feeding the channel.

### Events

| Kind | Meaning |
|---|---|
| `EventStart` | emitted once, before the first leg |
| `EventTextDelta` | fragment of answer text |
| `EventReasoningDelta` | fragment of thinking text |
| `EventToolCallStart` | a tool-call slot opened (`Index`) |
| `EventToolCallDelta` | raw argument JSON fragment |
| `EventToolCallEnd` | the completed call, in `ToolCall` |
| `EventLegFailed` | the chain is advancing; `Attempt` says why |
| `EventDone` | terminal: `stop`, `length` or `tool_use` |
| `EventError` | terminal: `error` or `aborted` |

## Result

`AssistantMessage` is what both `Ask` and `Result` return.

- `StopReason` — normalized: `stop`, `length`, `tool_use`, `error`, `aborted`
- `FinishReason` — the winning provider's own string, verbatim, never normalized
- `ErrorMessage` — set on failure, and names **every** failed leg
- `Attempts` — every leg tried, in order, with its own usage and error
- `Usage` — tokens for the winning leg only

```go
msg := client.Ask(ctx, req)
switch msg.StopReason {
case smolllm.StopReasonStop:
    fmt.Println(msg.Content)
case smolllm.StopReasonToolUse:
    for _, call := range msg.ToolCalls {
        dispatch(call)
    }
case smolllm.StopReasonLength:
    fmt.Println(msg.Content, "(truncated)")
case smolllm.StopReasonError, smolllm.StopReasonAborted:
    log.Printf("%s: %s", msg.StopReason, msg.ErrorMessage)
}
```

`StopReason` normalizes; `FinishReason` does not. Everything that is not
`length` or a tool call maps to `StopReasonStop`, **including `content_filter`
and provider-specific refusal strings**. A filtered response is therefore
indistinguishable from a normal one on `StopReason` alone — read `FinishReason`
to detect it. Nothing is lost: the provider's own string is always there.

`Usage` stops at tokens: `Input` excludes cached tokens, which are reported
separately as `CacheRead`; `Output` includes `Reasoning`; `Total` is
`Input + Output + CacheRead`. `Estimated` marks a heuristic count. There are no
cost fields — pricing is the caller's concern.

Every attempt is on `Attempts`, and `WithHook(func(smolllm.Attempt))` delivers
the same records live as each attempt finishes, on success as well as failure —
which is what a long-running server wants for its token ledger.

## Model Format

- `provider/model` — provider resolves base URL and API key from the table/env by name
- `model` (bare, no `/`) — no provider: base URL and API key must come from
  `WithProvider(BareProvider, ...)` explicitly (no env fallback); provider
  surfaces as `""` in usage, hooks, and logs
- Comma-separated specs form a fallback chain and may mix both forms
- Everything after the first `/` is the wire model name, verbatim:
  `openrouter/~deepseek/x` sends model `~deepseek/x`

Reasoning effort is set with `WithReasoningEffort` and applies to every leg.

## Credentials

`WithProvider` configures one provider; `WithDefaultProvider` covers the rest.
Per leg the precedence is `WithProvider` > `WithDefaultProvider` > environment >
provider table. Fields left empty fall through rather than masking a weaker tier.

```go
client := smolllm.New(
    smolllm.WithModel("deepseek/deepseek-v4-flash,groq/llama-3.3-70b"),
    smolllm.WithProvider("deepseek", smolllm.ProviderConfig{
        BaseURL: "https://api.deepseek.com",
        APIKey:  "sk-one,sk-two", // comma lists are balanced across calls
        Headers: map[string]string{"X-Tenant": "acme"},
    }),
    smolllm.WithDefaultProvider(smolllm.ProviderConfig{APIKey: "sk-shared"}),
)
```

Each `Client` owns its balancer, so clients built for different model roles never
share key rotation state.

## CLI

```
go run ./cmd/cli --model gemini/gemini-2.0-flash "Say hello world"
go run ./cmd/cli --model openai/gpt-5 --reasoning-effort medium "Say hello"
```

Flags:
- `--stream` stream deltas instead of waiting for completion
- `--system` inject system message
- `--image` attach image path or data URL (repeatable)
- `--temperature` control sampling randomness `[0,2]`
- `--top-p` control nucleus sampling cutoff `[0,1]`
- `--reasoning-effort` pass `none|minimal|low|medium|high|xhigh` to compatible providers
- `--timeout` override default `120s`
- `--strip-backticks` remove enclosing markdown fences

## Env Layout

- `SMOLLLM_MODEL` fallback when no `WithModel`
- `${PROVIDER}_API_KEY` comma list allowed
- `${PROVIDER}_BASE_URL` optional override, matches provider slug (hyphen → underscore)
- Bare models (no `provider/` prefix) never read env — explicit options only
- `LOG_LEVEL` optional (`DEBUG`, `INFO`, `WARN`, `ERROR`)

## Timeouts and Retries

`WithTimeout` bounds the **whole call**: every retry, every fallback leg, the
backoff waits between them, and the consumption of the stream. Zero disables the
bound. Default 600s (the CLI passes 120s).

`WithMaxRetries` caps the attempts against one model before the chain advances.

Failures are classified, and the disposition decides what happens next:

| Failure | Disposition |
|---|---|
| 400, 413, 422 | abort — the request shape is wrong for every leg |
| 401, 403, 404, 429 | advance — leg-local credentials, catalogue or quota |
| 500, 502, 503, 504, 529 | retry with backoff, then advance |
| connection, DNS, TLS, EOF | advance |
| whole-call deadline exceeded | abort, terminal `error` |
| caller cancelled or `Close()` | abort, terminal `aborted` |
| empty answer, truncation, `MinOutputTokens` | advance |

`Classify` is exported so a caller can reuse the same policy.

## Features

- key/base-url load balancing with usage tracking
- image prompts via `WithImagePaths`
- markdown fence stripping via `WithBacktickRemoval`
- fail-fast validation via `Client.Validate` (invoked automatically by the CLI)
- per-attempt telemetry via `WithHook(func(smolllm.Attempt))`
- raw request fields the library does not model via `WithExtraBody`

## Escape Hatch

`WithExtraBody` merges raw request fields into the payload last, so they win over
library defaults — including over typed `Request.Tools`:

```go
msg := client.Ask(ctx, req,
    smolllm.WithExtraBody(map[string]any{"response_format": map[string]any{"type": "json_object"}}),
)
```

The fields the library reads back — `stream`, `stream_options`, `messages`,
`model` — are rejected.

## Tool Calling

Declare tools on the `Request`. The JSON Schema in `Parameters` is passed through
untouched: smolllm never inspects or repairs it, and never executes a tool.

```go
req := smolllm.RequestFromString("weather in Paris?")
req.Tools = []smolllm.Tool{{
    Name:        "get_weather",
    Description: "Look up the current weather for a city",
    Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
}}

msg := client.Ask(ctx, req)
for _, call := range msg.ToolCalls {
    result := dispatch(call.Function.Name, call.Function.Arguments) // arguments = raw JSON text
    req.Messages = append(req.Messages,
        smolllm.AssistantToolCalls(msg.Content, msg.ToolCalls),
        smolllm.ToolResult(call.ID, result),
    )
}
```

When streaming, argument fragments arrive as `EventToolCallDelta` and the
complete call arrives on `EventToolCallEnd`. This diverges from the Python and
Rust ports, which expose tool calls only after the stream ends.

Notes:

- `FinishReason` is verbatim. Gemini reports `stop` while returning tool calls,
  so key on `StopReason == StopReasonToolUse`, which accounts for that.
- Unknown provider keys on a call (e.g. Gemini's thought signature) are kept and
  replayed, so multi-turn tool loops stay lossless.
- A model that rejects `tools` fails its leg and the fallback chain advances; a
  model that ignores them answers in prose.

## Tests

```
make test
```
