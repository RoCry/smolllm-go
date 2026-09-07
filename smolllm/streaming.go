package smolllm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"time"
)

// delta is a single streamed fragment: answer text, thinking text, or both.
type delta struct {
	Content   string
	Reasoning string
}

// IsEmpty reports whether the fragment carries neither content nor reasoning.
func (d delta) IsEmpty() bool { return d.Content == "" && d.Reasoning == "" }

type streamDelta struct {
	Content          *string         `json:"content"`
	ReasoningContent *string         `json:"reasoning_content"` // DeepSeek, vLLM, LiteLLM
	Reasoning        *string         `json:"reasoning"`         // Ollama
	ToolCalls        []toolCallDelta `json:"tool_calls"`
}

type streamChoice struct {
	Delta        *streamDelta `json:"delta"`
	FinishReason *string      `json:"finish_reason"`
}

type streamChunk struct {
	Choices []streamChoice `json:"choices"`
	Usage   *usageChunk    `json:"usage"`
}

// legOutcome is what one leg's stream produced.
type legOutcome struct {
	usage        reportedUsage
	finishReason string
	toolCalls    []ToolCall
	// toolIndexes are the provider slot numbers of toolCalls, in the same order,
	// so terminal tool-call events match the streamed fragments.
	toolIndexes []int
	ttft        time.Duration
	err         error
}

// streamSink receives what the parser pulls off the wire. The chain implements
// it to turn fragments into events and accumulate the assistant turn.
type streamSink interface {
	// text records answer or thinking text.
	text(fragment delta)
	// toolFragment records one streamed tool-call fragment.
	toolFragment(fragment toolCallFragment)
	// toolAccumulator is where assembled tool calls are built. The sink owns it
	// so the accumulated turn and the streamed events cannot drift apart.
	toolAccumulator() *toolCallAccumulator
}

// consumeLegStream reads one leg's SSE body to completion, pushing everything it
// finds into sink. It runs on the chain's own goroutine, so it blocks rather
// than forwarding through a channel.
func consumeLegStream(
	ctx context.Context,
	logger *slog.Logger,
	body io.Reader,
	start time.Time,
	sink streamSink,
) legOutcome {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var (
		firstToken   time.Time
		filter       thinkTagFilter
		usage        reportedUsage
		finishReason string
		err          error
	)
	tools := sink.toolAccumulator()

Loop:
	for scanner.Scan() {
		select {
		case <-ctx.Done():
			err = ctx.Err()
			break Loop
		default:
		}

		var fragment delta
		fragment, err = parseChunkLine(logger, scanner.Text(), &usage, &finishReason, tools, sink)
		if err != nil {
			break
		}

		fragment = filter.Feed(fragment)
		if fragment.IsEmpty() {
			continue
		}
		if firstToken.IsZero() {
			firstToken = time.Now().UTC()
		}
		sink.text(fragment)
	}

	// Flush any text held back while a partial <think> tag was buffered.
	if final := filter.Flush(); !final.IsEmpty() {
		sink.text(final)
	}

	if err == nil {
		err = scanner.Err()
	}
	if err == nil && ctx.Err() != nil {
		err = ctx.Err()
	}

	return legOutcome{
		usage:        usage,
		finishReason: finishReason,
		toolCalls:    tools.result(),
		toolIndexes:  tools.sortedIndexes(),
		ttft:         computeTTFT(firstToken, start),
		err:          err,
	}
}

// parseChunkLine decodes one SSE line. usage, finishReason, tools and sink may
// each be nil when a caller only wants the text fragment.
func parseChunkLine(
	logger *slog.Logger,
	line string,
	usage *reportedUsage,
	finishReason *string,
	tools *toolCallAccumulator,
	sink streamSink,
) (delta, error) {
	empty := delta{Content: "", Reasoning: ""}

	trimmed := strings.TrimSpace(line)
	if trimmed == "" || trimmed == "data: [DONE]" || !strings.HasPrefix(trimmed, "data:") {
		return empty, nil
	}
	payload := strings.TrimSpace(trimmed[len("data:"):])
	if payload == "" {
		return empty, nil
	}

	var chunk streamChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		logger.Error("malformed streaming chunk", "error", err)
		return empty, fmt.Errorf("malformed streaming chunk: %w", err)
	}

	if usage != nil && chunk.Usage != nil {
		usage.usage = parseUsage(*chunk.Usage)
		usage.reported = true
	}
	if finishReason != nil && len(chunk.Choices) > 0 && chunk.Choices[0].FinishReason != nil {
		*finishReason = *chunk.Choices[0].FinishReason
	}

	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		logger.Debug("stream chunk missing delta")
		return empty, nil
	}

	streamed := chunk.Choices[0].Delta
	if tools != nil && len(streamed.ToolCalls) > 0 {
		var observe func(toolCallFragment)
		if sink != nil {
			observe = sink.toolFragment
		}
		tools.feed(streamed.ToolCalls, observe)
	}

	content := ""
	if streamed.Content != nil {
		content = *streamed.Content
	}
	reasoning := extractReasoning(streamed)
	if content == "" && reasoning == "" {
		return empty, nil
	}
	return delta{Content: content, Reasoning: reasoning}, nil
}

func extractReasoning(d *streamDelta) string {
	if d.ReasoningContent != nil && *d.ReasoningContent != "" {
		return *d.ReasoningContent
	}
	if d.Reasoning != nil && *d.Reasoning != "" {
		return *d.Reasoning
	}
	return ""
}

func computeTTFT(firstToken time.Time, start time.Time) time.Duration {
	if firstToken.IsZero() {
		return -1
	}
	ttft := firstToken.Sub(start)
	if ttft < 0 {
		return 0
	}
	return ttft
}
