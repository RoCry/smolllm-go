package smolllm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

type chatCompletionRequest struct {
	Messages []messagePayload `json:"messages"`
	Model    string           `json:"model"`
	Stream   bool             `json:"stream"`
}

type messagePayload struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

func buildRequestPayload(prompt Prompt, systemPrompt string, modelName string, providerName string, baseURL string, imagePaths []string) (string, []byte, int, error) {
	if strings.TrimSpace(baseURL) == "" {
		return "", nil, 0, fmt.Errorf("base URL not provided")
	}

	messages, err := composeMessages(prompt, systemPrompt, imagePaths)
	if err != nil {
		return "", nil, 0, err
	}

	payload := chatCompletionRequest{
		Messages: messages,
		Model:    modelName,
		Stream:   true,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", nil, 0, fmt.Errorf("encode request: %w", err)
	}

	url := buildRequestURL(baseURL, providerName)
	return url, body, estimateTokens(string(body)), nil
}

func composeMessages(prompt Prompt, systemPrompt string, imagePaths []string) ([]messagePayload, error) {
	if err := prompt.Validate(); err != nil {
		return nil, err
	}

	if len(imagePaths) > 0 && len(prompt.Messages) != 1 {
		return nil, fmt.Errorf("image paths only supported with single user prompt")
	}

	var messages []messagePayload
	if strings.TrimSpace(systemPrompt) != "" {
		messages = append(messages, messagePayload{
			Role:    string(RoleSystem),
			Content: systemPrompt,
		})
	}

	for idx, msg := range prompt.Messages {
		role := strings.TrimSpace(string(msg.Role))
		if role == "" {
			return nil, fmt.Errorf("prompt message #%d must include role", idx)
		}

		switch content := msg.Content.(type) {
		case string:
			if len(imagePaths) > 0 {
				if msg.Role != RoleUser {
					return nil, fmt.Errorf("image paths require user role")
				}
				contentParts := make([]map[string]interface{}, 0, len(imagePaths)+1)
				contentParts = append(contentParts, map[string]interface{}{
					"type": "text",
					"text": content,
				})
				for _, path := range imagePaths {
					dataURL, err := imagePathToData(path)
					if err != nil {
						return nil, err
					}
					contentParts = append(contentParts, map[string]interface{}{
						"type":      "image_url",
						"image_url": map[string]string{"url": dataURL},
					})
				}
				messages = append(messages, messagePayload{Role: role, Content: contentParts})
			} else {
				messages = append(messages, messagePayload{Role: role, Content: content})
			}
		default:
			if len(imagePaths) > 0 {
				return nil, fmt.Errorf("image paths cannot be combined with structured prompt content")
			}
			messages = append(messages, messagePayload{Role: role, Content: content})
		}
	}

	return messages, nil
}

func buildRequestURL(baseURL, providerName string) string {
	base := strings.TrimSpace(baseURL)
	switch providerName {
	case "anthropic":
		return strings.TrimRight(base, "/") + "/v1"
	case "gemini":
		return strings.TrimRight(base, "/") + "/v1beta/openai/chat/completions"
	default:
		if strings.HasSuffix(base, "#") {
			return strings.TrimSuffix(base, "#")
		}
		if strings.HasSuffix(base, "/") {
			return base + "chat/completions"
		}
		return base + "/v1/chat/completions"
	}
}

func imagePathToData(path string) (string, error) {
	trimmed := strings.TrimSpace(path)
	if trimmed == "" {
		return "", fmt.Errorf("image path cannot be empty")
	}
	if strings.HasPrefix(trimmed, "data:") {
		return trimmed, nil
	}
	content, err := os.ReadFile(trimmed)
	if err != nil {
		return "", fmt.Errorf("read image %q: %w", trimmed, err)
	}

	mimeType := mimeTypeFor(trimmed, content)
	if mimeType == "" {
		return "", fmt.Errorf("unable to determine mime type for %q", trimmed)
	}

	encoded := base64.StdEncoding.EncodeToString(content)
	return fmt.Sprintf("data:%s;base64,%s", mimeType, encoded), nil
}

func mimeTypeFor(path string, content []byte) string {
	if ext := strings.ToLower(filepath.Ext(path)); ext != "" {
		if m := mime.TypeByExtension(ext); m != "" {
			return m
		}
	}
	if len(content) > 0 {
		sample := content
		if len(sample) > 512 {
			sample = sample[:512]
		}
		if detected := http.DetectContentType(sample); detected != "" && detected != "application/octet-stream" {
			return detected
		}
	}
	return ""
}
