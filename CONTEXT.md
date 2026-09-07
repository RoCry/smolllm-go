# SmolLLM (Go)

Go port of smolllm: a minimal client for many LLM providers over the OpenAI-compatible wire protocol — one interface, API-key/endpoint balancing, model fallback. Shares its ubiquitous language with the Python lib (`../smolllm/CONTEXT.md` is the canonical copy; keep in sync **except** where an entry below records a deliberate divergence).

## Language

**Provider**:
A named OpenAI-compatible endpoint (e.g. `openai`, `groq`); credentials and base URL resolve from env by name. Bare model specs have no provider — identity surfaces as empty string.
_Avoid_: vendor, backend.

**Bare model**:
A model spec with no `provider/` prefix. It has no provider, so base URL and API key come from explicit options only and never from env; the provider surfaces as the empty string, which `BareProvider` names in code.
_Avoid_: treating a known provider name used alone (`gemini`) as a provider — it is a bare model.

**Model spec**:
The user-facing model string `provider/model`, or bare `model` (no `/`). Everything after the first `/` is an opaque model name and reaches the wire verbatim (`openrouter/~deepseek/x` → `~deepseek/x`). Comma-separated specs form a fallback chain; may mix both forms.
_Diverges from Python_: the Python glossary still documents an `!effort` suffix; in Go, reasoning effort is `WithReasoningEffort` only.

**Fallback chain**:
Ordered or weighted candidate models; on failure the call advances to the next candidate.
_Avoid_: confusing with retry.

**Retry**:
Re-attempt of the *same* model after a transient failure. Distinct from fallback (which switches models).

**Attempt**:
One try of one leg — a first try or a retry of the same model. Carries the leg identity, usage, timing and any `LegError`. Every attempt of a call is collected on the terminal message and delivered live to the request hook.

**Disposition**:
What the chain does about a leg failure: `retry` (same model, after backoff), `advance` (next model in the chain), `abort` (stop now). Named to keep **Retry** and **Fallback chain** distinct.
_Avoid_: calling an advance a retry.

**Stop reason**:
Normalized reason a turn ended: `stop`, `length`, `tool_use`, `error`, `aborted` (plus non-terminal `pending`). Derived from FinishReason and from failure classification; FinishReason keeps the provider's verbatim string alongside it.
_Avoid_: conflating with FinishReason.

**Balancer pair**:
One (API key, base URL) combination for a provider; the least-used pair is chosen per call.

**Provider config**:
Explicit `(base URL, API key, headers)` for one provider, supplied by the caller instead of env lookup. Applies only to legs of that provider; `WithDefaultProvider` covers legs with no entry of their own.

**Estimated usage**:
Token counts derived by heuristic when the provider omits usage; always marked (`~` prefix, `Estimated` flag). Reported usage excludes cached tokens from input and counts them separately; output includes reasoning tokens.

**Reasoning**:
Model thinking text, kept in a channel separate from content.
_Avoid_: mixing reasoning into content.

**FinishReason**:
Verbatim provider string explaining why generation ended; never normalized. Distinct from **Stop reason**, which normalizes it.

**Request hook**:
Per-attempt observation callback receiving an `Attempt`; the library's only telemetry surface. Fires live as each attempt finishes, on success as well as failure; the same attempts are collected on the terminal message.

**Escape hatch**:
A pass-through (`WithExtraBody`) letting callers set raw request fields the library does not model, merged last so the caller wins. The fields the library machinery reads back (`stream`, `stream_options`, `messages`, `model`) are rejected.
_Avoid_: raw options, extra params.

**Tool call**:
A provider-issued request to run a named function, surfaced verbatim as a `ToolCall` carrying the wire fields with the argument JSON as an opaque string. The caller executes it and replays the assistant and `tool` messages; the library runs no agentic loop. When streaming, argument fragments are forwarded as they arrive and the complete call arrives on `tool_call_end`. The library still never parses, validates or repairs the argument JSON.
_Diverges from Python_: Python and Rust expose calls only after the stream ends.
_Avoid_: function call, tool use, implying smolllm executes anything.
