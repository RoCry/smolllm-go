package smolllm

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

func stripBackticks(text string) string {
	if !strings.HasPrefix(text, "```") || !strings.HasSuffix(text, "```") {
		return text
	}
	lines := strings.Split(text, "\n")
	if len(lines) == 0 {
		return ""
	}
	if strings.HasPrefix(lines[0], "```") {
		lines = lines[1:]
	}
	if len(lines) == 0 {
		return ""
	}
	last := lines[len(lines)-1]
	if strings.HasSuffix(last, "```") {
		if last == "```" {
			lines = lines[:len(lines)-1]
		} else {
			lines[len(lines)-1] = strings.TrimSuffix(last, "```")
		}
	}
	return strings.Join(lines, "\n")
}

func estimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return len(text) / 4
}

func formatMetrics(modelName string, inputTokens, outputTokens int, total time.Duration, ttft time.Duration) string {
	totalTokens := inputTokens + outputTokens
	tokPerSec := 0
	if total > 0 && outputTokens > 0 {
		tokPerSec = int(float64(outputTokens) / total.Seconds())
	}

	var builder strings.Builder
	builder.WriteString("📊")
	builder.WriteString(modelName)
	builder.WriteString(" ")
	builder.WriteString(strconv.Itoa(totalTokens))
	builder.WriteString("tok (↑")
	builder.WriteString(strconv.Itoa(inputTokens))
	builder.WriteString(" ↓")
	builder.WriteString(strconv.Itoa(outputTokens))
	builder.WriteString(")")

	if ttft >= 0 {
		builder.WriteString(" | 🚀")
		builder.WriteString(formatDuration(ttft))
	}

	builder.WriteString(" | 🐎")
	builder.WriteString(strconv.Itoa(tokPerSec))
	builder.WriteString("tok/s | ⌛")
	builder.WriteString(formatDuration(total))
	return builder.String()
}

func formatDuration(d time.Duration) string {
	if d < 0 {
		panic("duration cannot be negative")
	}
	ms := float64(d) / float64(time.Millisecond)
	switch {
	case ms >= 1000:
		return fmt.Sprintf("%.1fs", ms/1000.0)
	case ms >= 100:
		return fmt.Sprintf("%.0fms", ms)
	case ms >= 10:
		return fmt.Sprintf("%.1fms", ms)
	default:
		return fmt.Sprintf("%.2fms", ms)
	}
}

// ---------- <think> tag extraction ----------

var thinkRE = regexp.MustCompile(`(?s)<think>(.*?)</think>`)

// extractThinkTags extracts all <think> blocks from text, returning
// (reasoning, cleanContent). Reasoning parts are joined by double newline.
func extractThinkTags(text string) (reasoning, content string) {
	var parts []string
	clean := thinkRE.ReplaceAllStringFunc(text, func(match string) string {
		sub := thinkRE.FindStringSubmatch(match)
		if len(sub) > 1 {
			parts = append(parts, strings.TrimSpace(sub[1]))
		}
		return ""
	})
	return strings.Join(parts, "\n\n"), strings.TrimSpace(clean)
}

// ThinkTagFilter is a stateful streaming filter that reclassifies content
// inside <think> tags as reasoning. Auto-disables if the backend already
// provides reasoning_content in the delta. Handles tag splits across chunk
// boundaries via buffering.
type ThinkTagFilter struct {
	insideThink bool
	buffer      string
	disabled    bool
}

const openTag = "<think>"
const closeTag = "</think>"

// Feed processes a StreamChunk, reclassifying <think> content as reasoning.
func (f *ThinkTagFilter) Feed(chunk StreamChunk) StreamChunk {
	// If backend already provides reasoning, pass through and disable.
	if chunk.Reasoning != "" {
		f.disabled = true
		return chunk
	}
	if f.disabled {
		return chunk
	}

	text := f.buffer + chunk.Content
	f.buffer = ""

	var reasoningParts []string
	var contentParts []string

	for text != "" {
		if f.insideThink {
			end := strings.Index(text, closeTag)
			if end == -1 {
				// Check for partial closing tag at end.
				if buffered := partialSuffix(text, closeTag); buffered > 0 {
					f.buffer = text[len(text)-buffered:]
					text = text[:len(text)-buffered]
				}
				if text != "" {
					reasoningParts = append(reasoningParts, text)
				}
				break
			}
			reasoningParts = append(reasoningParts, text[:end])
			text = text[end+len(closeTag):]
			f.insideThink = false
		} else {
			start := strings.Index(text, openTag)
			if start == -1 {
				// Check for partial opening tag at end.
				if buffered := partialSuffix(text, openTag); buffered > 0 {
					f.buffer = text[len(text)-buffered:]
					text = text[:len(text)-buffered]
				}
				if text != "" {
					contentParts = append(contentParts, text)
				}
				break
			}
			if start > 0 {
				contentParts = append(contentParts, text[:start])
			}
			text = text[start+len(openTag):]
			f.insideThink = true
		}
	}

	return StreamChunk{
		Content:   strings.Join(contentParts, ""),
		Reasoning: strings.Join(reasoningParts, ""),
	}
}

// Flush returns any buffered content at end of stream.
func (f *ThinkTagFilter) Flush() StreamChunk {
	if f.buffer == "" {
		return StreamChunk{}
	}
	buf := f.buffer
	f.buffer = ""
	if f.insideThink {
		return StreamChunk{Reasoning: buf}
	}
	return StreamChunk{Content: buf}
}

// partialSuffix returns the length of the longest proper suffix of text
// that is a prefix of tag, or 0 if none.
func partialSuffix(text, tag string) int {
	maxLen := len(tag) - 1
	if maxLen > len(text) {
		maxLen = len(text)
	}
	for i := maxLen; i > 0; i-- {
		if text[len(text)-i:] == tag[:i] {
			return i
		}
	}
	return 0
}
