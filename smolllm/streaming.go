package smolllm

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

type streamDelta struct {
	Content          *string `json:"content"`
	ReasoningContent *string `json:"reasoning_content"` // DeepSeek, vLLM, LiteLLM
	Reasoning        *string `json:"reasoning"`         // Ollama
}

type streamChoice struct {
	Delta *streamDelta `json:"delta"`
}

type streamChunk struct {
	Choices []streamChoice `json:"choices"`
}

func startStreamForwarder(
	logger *slog.Logger,
	reqCtx context.Context,
	resp *http.Response,
	call *preparedCall,
	cancel context.CancelFunc,
	start time.Time,
) (chan StreamChunk, chan streamCompletion) {
	chunks := make(chan StreamChunk)
	done := make(chan streamCompletion, 1)

	go func() {
		defer cancel()
		defer resp.Body.Close()
		defer close(chunks)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

		var (
			firstToken       time.Time
			contentBuilder   strings.Builder
			reasoningBuilder strings.Builder
			thinkFilter      ThinkTagFilter
			err              error
		)

	Loop:
		for scanner.Scan() {
			select {
			case <-reqCtx.Done():
				err = reqCtx.Err()
				break Loop
			default:
			}

			line := scanner.Text()
			var chunk StreamChunk
			chunk, err = processChunkLine(logger, line)
			if err != nil {
				break
			}

			chunk = thinkFilter.Feed(chunk)
			if chunk.IsEmpty() {
				continue
			}

			if firstToken.IsZero() {
				firstToken = time.Now().UTC()
			}

			contentBuilder.WriteString(chunk.Content)
			reasoningBuilder.WriteString(chunk.Reasoning)

			select {
			case <-reqCtx.Done():
				err = reqCtx.Err()
				break Loop
			case chunks <- chunk:
			}
		}

		// Flush any buffered think-tag content.
		if final := thinkFilter.Flush(); !final.IsEmpty() {
			contentBuilder.WriteString(final.Content)
			reasoningBuilder.WriteString(final.Reasoning)
			select {
			case <-reqCtx.Done():
			case chunks <- final:
			}
		}

		if err == nil {
			err = scanner.Err()
		}

		total := time.Since(start)
		ttft := computeTTFT(firstToken, start)

		completion := streamCompletion{
			err:       err,
			metrics:   nil,
			reasoning: reasoningBuilder.String(),
		}

		if err == nil || errors.Is(err, context.Canceled) {
			combined := contentBuilder.String() + reasoningBuilder.String()
			completion.metrics = &streamMetrics{
				modelName:    call.ModelName,
				inputTokens:  call.InputTokens,
				outputTokens: estimateTokens(combined),
				total:        total,
				ttft:         ttft,
			}
		}

		done <- completion
		close(done)
	}()

	return chunks, done
}

type consumeResult struct {
	content   string
	reasoning string
	ttft      time.Duration
}

func consumeStream(
	logger *slog.Logger,
	ctx context.Context,
	reader io.Reader,
	handler func(context.Context, string) error,
	start time.Time,
) (consumeResult, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var contentBuilder strings.Builder
	var reasoningBuilder strings.Builder
	var thinkFilter ThinkTagFilter
	var firstToken time.Time

	for scanner.Scan() {
		if ctx.Err() != nil {
			return consumeResult{}, ctx.Err()
		}

		line := scanner.Text()
		chunk, err := processChunkLine(logger, line)
		if err != nil {
			return consumeResult{}, err
		}

		chunk = thinkFilter.Feed(chunk)
		if chunk.IsEmpty() {
			continue
		}

		if firstToken.IsZero() {
			firstToken = time.Now().UTC()
		}

		if handler != nil {
			if err := handler(ctx, chunk.Content); err != nil {
				return consumeResult{}, err
			}
		}
		contentBuilder.WriteString(chunk.Content)
		reasoningBuilder.WriteString(chunk.Reasoning)
	}

	// Flush any buffered think-tag content.
	if final := thinkFilter.Flush(); !final.IsEmpty() {
		contentBuilder.WriteString(final.Content)
		reasoningBuilder.WriteString(final.Reasoning)
	}

	if err := scanner.Err(); err != nil {
		return consumeResult{}, err
	}

	content := strings.TrimSpace(contentBuilder.String())
	reasoning := strings.TrimSpace(reasoningBuilder.String())
	if content == "" && reasoning == "" {
		return consumeResult{}, fmt.Errorf("empty response from model")
	}

	ttft := computeTTFT(firstToken, start)
	return consumeResult{content: content, reasoning: reasoning, ttft: ttft}, nil
}

func processChunkLine(logger *slog.Logger, line string) (StreamChunk, error) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" || trimmed == "data: [DONE]" {
		return StreamChunk{}, nil
	}
	if !strings.HasPrefix(trimmed, "data:") {
		return StreamChunk{}, nil
	}
	payload := strings.TrimSpace(trimmed[len("data:"):])
	if payload == "" {
		return StreamChunk{}, nil
	}

	var chunk streamChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		logger.Error("malformed streaming chunk", "error", err)
		return StreamChunk{}, fmt.Errorf("malformed streaming chunk: %w", err)
	}

	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil {
		logger.Debug("stream chunk missing delta")
		return StreamChunk{}, nil
	}

	delta := chunk.Choices[0].Delta
	content := ""
	if delta.Content != nil {
		content = *delta.Content
	}
	reasoning := extractReasoning(delta)

	if content == "" && reasoning == "" {
		return StreamChunk{}, nil
	}
	return StreamChunk{Content: content, Reasoning: reasoning}, nil
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
