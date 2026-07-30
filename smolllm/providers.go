package smolllm

import (
	"fmt"
	"strings"
)

type provider struct {
	Name    string
	BaseURL string
}

const providerOllama = "ollama"

var providers = map[string]provider{
	"aihubmix":                {Name: "aihubmix", BaseURL: "https://aihubmix.com"},
	"anthropic":               {Name: "anthropic", BaseURL: "https://api.anthropic.com/"},
	"azure-openai":            {Name: "azure-openai", BaseURL: ""},
	"baichuan":                {Name: "baichuan", BaseURL: "https://api.baichuan-ai.com"},
	"baidu-cloud":             {Name: "baidu-cloud", BaseURL: "https://qianfan.baidubce.com/v2/"},
	"dashscope":               {Name: "dashscope", BaseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1/"},
	"deepseek":                {Name: "deepseek", BaseURL: "https://api.deepseek.com"},
	"dmxapi":                  {Name: "dmxapi", BaseURL: "https://www.dmxapi.cn"},
	"doubao":                  {Name: "doubao", BaseURL: "https://ark.cn-beijing.volces.com/api/v3/"},
	"fireworks":               {Name: "fireworks", BaseURL: "https://api.fireworks.ai/inference"},
	"gemini":                  {Name: "gemini", BaseURL: "https://generativelanguage.googleapis.com"},
	"gitee-ai":                {Name: "gitee-ai", BaseURL: "https://ai.gitee.com"},
	"github":                  {Name: "github", BaseURL: "https://models.inference.ai.azure.com/"},
	"graphrag-kylin-mountain": {Name: "graphrag-kylin-mountain", BaseURL: ""},
	"grok":                    {Name: "grok", BaseURL: "https://api.x.ai"},
	"groq":                    {Name: "groq", BaseURL: "https://api.groq.com/openai"},
	"hunyuan":                 {Name: "hunyuan", BaseURL: "https://api.hunyuan.cloud.tencent.com"},
	"hyperbolic":              {Name: "hyperbolic", BaseURL: "https://api.hyperbolic.xyz"},
	"infini":                  {Name: "infini", BaseURL: "https://cloud.infini-ai.com/maas"},
	"jina":                    {Name: "jina", BaseURL: "https://api.jina.ai"},
	"lmstudio":                {Name: "lmstudio", BaseURL: "http://localhost:1234"},
	"minimax":                 {Name: "minimax", BaseURL: "https://api.minimax.chat/v1/"},
	"mistral":                 {Name: "mistral", BaseURL: "https://api.mistral.ai"},
	"modelscope":              {Name: "modelscope", BaseURL: "https://api-inference.modelscope.cn/v1/"},
	"moonshot":                {Name: "moonshot", BaseURL: "https://api.moonshot.cn"},
	"nvidia":                  {Name: "nvidia", BaseURL: "https://integrate.api.nvidia.com"},
	"o3":                      {Name: "o3", BaseURL: "https://api.o3.fan"},
	"ocoolai":                 {Name: "ocoolai", BaseURL: "https://api.ocoolai.com"},
	providerOllama:            {Name: providerOllama, BaseURL: "http://localhost:11434"},
	"openai":                  {Name: "openai", BaseURL: "https://api.openai.com"},
	"openrouter":              {Name: "openrouter", BaseURL: "https://openrouter.ai/api/v1/"},
	"perplexity":              {Name: "perplexity", BaseURL: "https://api.perplexity.ai/"},
	"ppio":                    {Name: "ppio", BaseURL: "https://api.ppinfra.com/v3/openai"},
	"silicon":                 {Name: "silicon", BaseURL: "https://api.siliconflow.cn"},
	"stepfun":                 {Name: "stepfun", BaseURL: "https://api.stepfun.com"},
	"tencent-cloud-ti":        {Name: "tencent-cloud-ti", BaseURL: "https://api.lkeap.cloud.tencent.com"},
	"together":                {Name: "together", BaseURL: "https://api.together.xyz"},
	"xirang":                  {Name: "xirang", BaseURL: "https://wishub-x1.ctyun.cn"},
	"yi":                      {Name: "yi", BaseURL: "https://api.lingyiwanwu.com"},
	"zhinao":                  {Name: "zhinao", BaseURL: "https://api.360.cn"},
	"zhipu":                   {Name: "zhipu", BaseURL: "https://open.bigmodel.cn/api/paas/v4/"},
}

func parseModelSpec(spec string) (string, *string) {
	model, effort, ok := strings.Cut(spec, "!")
	model = strings.TrimSpace(model)
	if !ok {
		return model, nil
	}
	effort = strings.TrimSpace(effort)
	if effort == "" {
		return model, nil
	}
	return model, &effort
}

// parseModelString splits a model string on the first "/" into provider and
// model name. Unknown prefixed providers still work when a base URL is supplied
// explicitly or via env. A string without "/" is a bare model name with no
// provider (Name == ""): base URL and API key must then be supplied explicitly.
func parseModelString(model string) (provider, string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return provider{}, "", fmt.Errorf("model string must not be empty")
	}

	if !strings.Contains(model, "/") {
		return provider{Name: "", BaseURL: ""}, model, nil
	}

	parts := strings.SplitN(model, "/", 2)
	providerName, modelName := parts[0], strings.TrimSpace(parts[1])
	if providerName == "" {
		return provider{}, "", fmt.Errorf("provider name missing in model string %q", model)
	}
	if modelName == "" {
		return provider{}, "", fmt.Errorf("model name missing for provider %q", providerName)
	}

	prov, ok := providers[providerName]
	if !ok {
		prov = provider{Name: providerName, BaseURL: ""}
	}

	return prov, modelName, nil
}
