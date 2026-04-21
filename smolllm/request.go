package smolllm

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	openai "github.com/openai/openai-go/v3"
)

var versionSuffixRE = regexp.MustCompile(`/v\d+$`)

type chatCompletionRequest struct {
	Messages        []openai.ChatCompletionMessageParamUnion `json:"messages"`
	Model           string                                   `json:"model"`
	Stream          bool                                     `json:"stream"`
	Temperature     *float64                                 `json:"temperature,omitempty"`
	TopP            *float64                                 `json:"top_p,omitempty"`
	ReasoningEffort *string                                  `json:"reasoning_effort,omitempty"`
}

func buildRequestPayload(
	prompt Prompt,
	systemPrompt string,
	modelName string,
	providerName string,
	baseURL string,
	imagePaths []string,
	temperature *float64,
	topP *float64,
	reasoningEffort *string,
) (string, []byte, int, error) {
	if strings.TrimSpace(baseURL) == "" {
		return "", nil, 0, fmt.Errorf("base URL not provided")
	}

	messages, err := composeMessages(prompt, systemPrompt, imagePaths)
	if err != nil {
		return "", nil, 0, err
	}

	if temperature != nil {
		if math.IsNaN(*temperature) || *temperature < 0 || *temperature > 2 {
			return "", nil, 0, fmt.Errorf("temperature %f must be between 0 and 2 inclusive", *temperature)
		}
	}
	if topP != nil {
		if math.IsNaN(*topP) || *topP < 0 || *topP > 1 {
			return "", nil, 0, fmt.Errorf("top_p %f must be between 0 and 1 inclusive", *topP)
		}
	}
	if reasoningEffort != nil {
		v := strings.TrimSpace(*reasoningEffort)
		if v == "" {
			return "", nil, 0, fmt.Errorf("reasoning_effort must not be empty")
		}
	}

	payload := chatCompletionRequest{
		Messages:        messages,
		Model:           modelName,
		Stream:          true,
		Temperature:     temperature,
		TopP:            topP,
		ReasoningEffort: reasoningEffort,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", nil, 0, fmt.Errorf("encode request: %w", err)
	}

	url := buildRequestURL(baseURL, providerName)
	return url, body, estimateTokens(string(body)), nil
}

func composeMessages(
	prompt Prompt, systemPrompt string, imagePaths []string,
) ([]openai.ChatCompletionMessageParamUnion, error) {
	if err := prompt.Validate(); err != nil {
		return nil, err
	}

	if len(imagePaths) > 0 {
		// Count user messages to check if we have exactly one
		userMsgCount := 0
		for _, msg := range prompt.Messages {
			if role, ok := messageRole(msg); ok && strings.EqualFold(role, "user") {
				userMsgCount++
			}
		}
		if userMsgCount != 1 {
			return nil, fmt.Errorf("image paths require exactly one user message (system messages are allowed)")
		}
	}

	need := len(prompt.Messages)
	if strings.TrimSpace(systemPrompt) != "" {
		need++
	}

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, need)
	if strings.TrimSpace(systemPrompt) != "" {
		sys := openai.SystemMessage(systemPrompt)
		ensureRole(&sys)
		messages = append(messages, sys)
	}

	for idx, msg := range prompt.Messages {
		if len(imagePaths) > 0 {
			role, ok := messageRole(msg)
			if !ok {
				return nil, fmt.Errorf("prompt message #%d must include role", idx)
			}

			// Skip system messages when images are present - they're handled separately
			if strings.EqualFold(role, "system") {
				ensureRole(&msg)
				messages = append(messages, msg)
				continue
			}

			// Only user messages can have images attached
			if !strings.EqualFold(role, "user") {
				return nil, fmt.Errorf("image paths require user role")
			}

			text, err := messageTextContent(msg)
			if err != nil {
				return nil, fmt.Errorf("image paths require textual content: %w", err)
			}

			parts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(imagePaths)+1)
			parts = append(parts, openai.TextContentPart(text))
			for _, path := range imagePaths {
				dataURL, err := imagePathToData(path)
				if err != nil {
					return nil, err
				}
				parts = append(parts, openai.ImageContentPart(openai.ChatCompletionContentPartImageImageURLParam{
					URL:    dataURL,
					Detail: "auto",
				}))
			}

			u := openai.UserMessage(parts)
			ensureRole(&u)
			messages = append(messages, u)
			continue
		}

		if _, ok := messageRole(msg); !ok {
			return nil, fmt.Errorf("prompt message #%d must include role", idx)
		}

		ensureRole(&msg)
		messages = append(messages, msg)
	}

	return messages, nil
}

func messageTextContent(msg openai.ChatCompletionMessageParamUnion) (string, error) {
	contentAny := msg.GetContent().AsAny()
	if contentAny == nil {
		return "", fmt.Errorf("content missing")
	}

	switch v := contentAny.(type) {
	case *string:
		if v == nil {
			return "", fmt.Errorf("content missing")
		}
		return *v, nil
	default:
		return "", fmt.Errorf("content has type %T", v)
	}
}

// hasVersionSuffix checks if the URL (after stripping trailing "/") ends with
// a version path segment like /v1, /v2, /v3, etc.
func hasVersionSuffix(url string) bool {
	return versionSuffixRE.MatchString(strings.TrimRight(url, "/"))
}

// resolveEndpointURL builds the full API URL for a given endpoint (e.g. "chat/completions", "embeddings").
// It honors trailing-# (literal URL), trailing-/ (append endpoint), version-suffix, and provider-specific rules.
func resolveEndpointURL(baseURL, providerName, endpoint string) string {
	base := strings.TrimSpace(baseURL)
	switch providerName {
	case "anthropic":
		stripped := strings.TrimRight(base, "/")
		if hasVersionSuffix(stripped) {
			return stripped + "/" + endpoint
		}
		return stripped + "/v1/" + endpoint
	case "gemini":
		stripped := strings.TrimRight(base, "/")
		if hasVersionSuffix(stripped) {
			return stripped + "/" + endpoint
		}
		return stripped + "/v1beta/openai/" + endpoint
	default:
		if strings.HasSuffix(base, "#") {
			return strings.TrimSuffix(base, "#")
		}
		if strings.HasSuffix(base, "/") {
			return base + endpoint
		}
		if hasVersionSuffix(base) {
			return base + "/" + endpoint
		}
		return base + "/v1/" + endpoint
	}
}

func buildRequestURL(baseURL, providerName string) string {
	return resolveEndpointURL(baseURL, providerName, "chat/completions")
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
