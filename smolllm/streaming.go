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
	Content *string `json:"content"`
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
) (chan string, chan error) {
	chunks := make(chan string)
	done := make(chan error, 1)

	go func() {
		defer cancel()
		defer resp.Body.Close()
		defer close(chunks)

		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

		var (
			firstToken time.Time
			builder    strings.Builder
			err        error
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
			var delta string
			delta, err = processChunkLine(logger, line)
			if err != nil {
				break
			}
			if delta == "" {
				continue
			}

			if firstToken.IsZero() {
				firstToken = time.Now().UTC()
			}

			builder.WriteString(delta)

			select {
			case <-reqCtx.Done():
				err = reqCtx.Err()
				break Loop
			case chunks <- delta:
			}
		}

		if err == nil {
			err = scanner.Err()
		}

		total := time.Since(start)
		ttft := computeTTFT(firstToken, start)

		if err == nil || errors.Is(err, context.Canceled) {
			outputTokens := estimateTokens(builder.String())
			logger.Info(
				formatMetrics(call.ModelName, call.InputTokens, outputTokens, total, ttft),
				"model", call.ModelName,
			)
		}

		done <- err
		close(done)
	}()

	return chunks, done
}

func consumeStream(
	logger *slog.Logger,
	ctx context.Context,
	reader io.Reader,
	handler func(context.Context, string) error,
	start time.Time,
) (string, time.Duration, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	var builder strings.Builder
	var firstToken time.Time

	for scanner.Scan() {
		if ctx.Err() != nil {
			return "", -1, ctx.Err()
		}

		line := scanner.Text()
		delta, err := processChunkLine(logger, line)
		if err != nil {
			return "", -1, err
		}
		if delta == "" {
			continue
		}

		if firstToken.IsZero() {
			firstToken = time.Now().UTC()
		}

		if handler != nil {
			if err := handler(ctx, delta); err != nil {
				return "", -1, err
			}
		}
		builder.WriteString(delta)
	}

	if err := scanner.Err(); err != nil {
		return "", -1, err
	}

	result := strings.TrimSpace(builder.String())
	if result == "" {
		return "", -1, fmt.Errorf("empty response from model")
	}

	ttft := computeTTFT(firstToken, start)
	return result, ttft, nil
}

func processChunkLine(logger *slog.Logger, line string) (string, error) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" || trimmed == "data: [DONE]" {
		return "", nil
	}
	if !strings.HasPrefix(trimmed, "data:") {
		return "", nil
	}
	payload := strings.TrimSpace(trimmed[len("data:"):])
	if payload == "" {
		return "", nil
	}

	var chunk streamChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		logger.Error("malformed streaming chunk", "error", err)
		return "", fmt.Errorf("malformed streaming chunk: %w", err)
	}

	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil || chunk.Choices[0].Delta.Content == nil {
		logger.Debug("stream chunk missing content")
		return "", nil
	}
	return *chunk.Choices[0].Delta.Content, nil
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
