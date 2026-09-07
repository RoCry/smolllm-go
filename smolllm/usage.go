package smolllm

// usageChunk is the provider's usage frame. Providers disagree on where cached
// and reasoning counts live, so every known spelling is read here.
type usageChunk struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	// PromptCacheHitTokens is DeepSeek's spelling of the cached prompt count.
	PromptCacheHitTokens    int                      `json:"prompt_cache_hit_tokens"`
	PromptTokensDetails     *promptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details"`
}

// promptTokensDetails is the OpenAI spelling of the cached prompt count.
type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

type completionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// reportedUsage tracks whether the provider sent a usage frame at all, so an
// absent one falls back to estimation instead of reporting zeros as fact.
type reportedUsage struct {
	usage    Usage
	reported bool
}

// cachedTokens picks whichever spelling of the cached prompt count the provider
// used. DeepSeek's dedicated field wins when both are present.
func (u usageChunk) cachedTokens() int {
	if u.PromptCacheHitTokens > 0 {
		return u.PromptCacheHitTokens
	}
	if u.PromptTokensDetails != nil {
		return u.PromptTokensDetails.CachedTokens
	}
	return 0
}

func (u usageChunk) reasoningTokens() int {
	if u.CompletionTokensDetails != nil {
		return u.CompletionTokensDetails.ReasoningTokens
	}
	return 0
}

// parseUsage converts a provider usage frame into a Usage. Providers report
// prompt_tokens inclusive of cached tokens, so the cached count is subtracted
// back out to honour the rule that Input excludes CacheRead.
func parseUsage(chunk usageChunk) Usage {
	cacheRead := chunk.cachedTokens()
	input := chunk.PromptTokens - cacheRead
	if input < 0 {
		// A provider whose counts disagree must not produce a negative Input.
		input = 0
	}
	return newUsage(input, chunk.CompletionTokens, cacheRead, chunk.reasoningTokens(), false)
}

// estimateUsage builds a heuristic Usage for a provider that sent no usage
// frame. It is always marked Estimated.
func estimateUsage(inputTokens int, outputText string) Usage {
	return newUsage(inputTokens, estimateTokens(outputText), 0, 0, true)
}
