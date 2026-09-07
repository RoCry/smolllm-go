// Package smolllm provides a tiny Go client for OpenAI-compatible chat
// completions with multi-provider routing.
//
// Build a [Client] per configured model role and share it across callers.
// Chat never reports an operational failure as a Go error: [Client.Stream] and
// [Client.Ask] return an [AssistantMessage] whose StopReason says how the call
// ended, and whose ErrorMessage names every leg that failed. Programmer errors
// such as a nil context panic instead. [Client.Embed] keeps an error return,
// because it is not a streaming surface.
package smolllm
