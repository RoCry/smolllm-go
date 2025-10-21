package smolllm

import (
	"fmt"
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
