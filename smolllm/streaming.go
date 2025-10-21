package smolllm

import (
	"encoding/json"
	"fmt"
	"strings"
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

func processChunkLine(line string) (string, error) {
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
		logger.Error("malformed streaming chunk", "payload", payload, "err", err)
		return "", fmt.Errorf("malformed streaming chunk: %w", err)
	}

	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil || chunk.Choices[0].Delta.Content == nil {
		return "", nil
	}
	return *chunk.Choices[0].Delta.Content, nil
}
