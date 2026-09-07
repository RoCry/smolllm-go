# AGENTS.md

Guidance for coding agents working in this repository.

## Project Overview

smolllm-go is the Go port of smolllm (Python, sibling repo `../smolllm`): a minimal client for many LLM providers over the OpenAI-compatible wire protocol — one interface (`Client.Ask`/`Stream`/`Embed`/`Validate`), API-key/endpoint load balancing, model fallback chains, token metering. Consumed by smolllm-server (OpenAI-compatible proxy, sibling repo).

`Client` is the entry point: build one per configured model role and share it across tenants. Each `Client` owns its balancer, so key rotation state never leaks between roles. Package-level `Ask`/`Stream`/`Embed`/`Validate` delegate to one shared client.

Chat is **never-throw**: `Stream` and `Ask` report operational failure on the returned `AssistantMessage` (`StopReason`, `ErrorMessage`), not as a Go error. Only programmer errors panic (nil context, option misuse). `Embed` keeps its error return — it is not a streaming surface.

## Design Philosophy

**Extreme minimalism.** Scope is frozen at chat + embeddings over the OpenAI-compat wire: no new modalities, no native provider transports, and no new API surface without a consumer to exercise it. Tool calling and the `WithExtraBody` escape hatch follow the design recorded in the Python repo; JSON mode is absorbed by the escape hatch and needs no code.

Canonical doctrine lives in the Python repo — read before proposing features. It is canonical **except where the Divergences section below says otherwise**:
- `../smolllm/docs/adr/0001-extreme-minimalism.md` — scope freeze
- `../smolllm/docs/adr/0002-token-only-accounting.md` — usage stops at tokens; cost is the caller's concern
- `../smolllm/docs/DEFERRED.md` — recorded designs (incl. Go specifics) awaiting a real use case

## Divergences from the Python doctrine

Three deliberate departures, all from v0.3, all driven by the same consumer: the **agentiu agent loop**. Python and Rust are unchanged — do not edit the Python repo to match.

**Streamed tool-call argument fragments.** `DEFERRED.md` records "accumulate tool-call deltas internally, expose after stream end; no partial-JSON pushes to handlers". Go reverses this: fragments are pushed as `EventToolCallDelta` while they stream, and the assembled call arrives on `EventToolCallEnd`. The agent loop renders tool arguments as they arrive. The library still never parses or repairs the JSON — it only forwards bytes — so the "no inspection" half of the doctrine holds. Anything relying on "a consumer never sees partial argument JSON" is wrong for Go.

**No `!effort` model-spec suffix.** Both glossaries used to document `provider/model[!effort]`. Go dropped the suffix: `WithReasoningEffort` is the only path, and everything after the first `/` is an opaque model name. This lets a model name contain any punctuation, which the agent loop needs for OpenRouter-style specs. The Python glossary still documents the suffix; the Go `CONTEXT.md` marks the divergence rather than letting the two drift silently.

**All-legs error reporting.** v0.2 surfaced only the last leg's error. Go now classifies each failure (`Classify`, `Disposition`) and joins every failed leg into `AssistantMessage.ErrorMessage`, with the full list on `Attempts`. `DEFERRED.md` records the same defect in Rust, so Go leads the family here rather than diverging on purpose; Rust and Python can follow when someone ports it.

Domain glossary: [CONTEXT.md](CONTEXT.md).

## Development

- `make test` — `go test -v -race ./...` (offline: providers faked with `httptest`, no API keys needed)
- `make lint` — golangci-lint v2, strict profile. The pinned v2.11.4 binary cannot read Go 1.27 export data, so run it under an older toolchain: `GOTOOLCHAIN=go1.26.2 make lint`. Note also that `.golangci.yml` still uses the v1 keys `output.formats.colored-line-number` and top-level `linters-settings`, which v2 silently ignores (`golangci-lint config verify` reports both); the effective profile is therefore the defaults plus the enabled linter list, not the settings written in the file.
- Provider map is hand-maintained in `smolllm/providers.go` (the Python repo's `providers.json` is generated; sync manually — no generator yet)
