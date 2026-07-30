package smolllm

import (
	"fmt"
	"strings"
)

type provider struct {
	Name         string
	BaseURL      string
	DefaultModel string
}

const providerOllama = "ollama"

var providers = map[string]provider{
	"aihubmix":     {Name: "aihubmix", BaseURL: "https://aihubmix.com", DefaultModel: ""},
	"anthropic":    {Name: "anthropic", BaseURL: "https://api.anthropic.com/", DefaultModel: ""},
	"azure-openai": {Name: "azure-openai", BaseURL: "", DefaultModel: ""},
	"baichuan":     {Name: "baichuan", BaseURL: "https://api.baichuan-ai.com", DefaultModel: ""},
	"baidu-cloud":  {Name: "baidu-cloud", BaseURL: "https://qianfan.baidubce.com/v2/", DefaultModel: ""},
	"dashscope": {
		Name: "dashscope", BaseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1/", DefaultModel: "",
	},
	"deepseek":  {Name: "deepseek", BaseURL: "https://api.deepseek.com", DefaultModel: ""},
	"dmxapi":    {Name: "dmxapi", BaseURL: "https://www.dmxapi.cn", DefaultModel: ""},
	"doubao":    {Name: "doubao", BaseURL: "https://ark.cn-beijing.volces.com/api/v3/", DefaultModel: ""},
	"fireworks": {Name: "fireworks", BaseURL: "https://api.fireworks.ai/inference", DefaultModel: ""},
	"gemini": {
		Name: "gemini", BaseURL: "https://generativelanguage.googleapis.com", DefaultModel: "gemini-2.0-flash",
	},
	"gitee-ai":                {Name: "gitee-ai", BaseURL: "https://ai.gitee.com", DefaultModel: ""},
	"github":                  {Name: "github", BaseURL: "https://models.inference.ai.azure.com/", DefaultModel: ""},
	"graphrag-kylin-mountain": {Name: "graphrag-kylin-mountain", BaseURL: "", DefaultModel: ""},
	"grok":                    {Name: "grok", BaseURL: "https://api.x.ai", DefaultModel: ""},
	"groq":                    {Name: "groq", BaseURL: "https://api.groq.com/openai", DefaultModel: ""},
	"hunyuan":                 {Name: "hunyuan", BaseURL: "https://api.hunyuan.cloud.tencent.com", DefaultModel: ""},
	"hyperbolic":              {Name: "hyperbolic", BaseURL: "https://api.hyperbolic.xyz", DefaultModel: ""},
	"infini":                  {Name: "infini", BaseURL: "https://cloud.infini-ai.com/maas", DefaultModel: ""},
	"jina":                    {Name: "jina", BaseURL: "https://api.jina.ai", DefaultModel: ""},
	"lmstudio":                {Name: "lmstudio", BaseURL: "http://localhost:1234", DefaultModel: ""},
	"minimax":                 {Name: "minimax", BaseURL: "https://api.minimax.chat/v1/", DefaultModel: ""},
	"mistral":                 {Name: "mistral", BaseURL: "https://api.mistral.ai", DefaultModel: ""},
	"modelscope":              {Name: "modelscope", BaseURL: "https://api-inference.modelscope.cn/v1/", DefaultModel: ""},
	"moonshot":                {Name: "moonshot", BaseURL: "https://api.moonshot.cn", DefaultModel: ""},
	"nvidia":                  {Name: "nvidia", BaseURL: "https://integrate.api.nvidia.com", DefaultModel: ""},
	"o3":                      {Name: "o3", BaseURL: "https://api.o3.fan", DefaultModel: ""},
	"ocoolai":                 {Name: "ocoolai", BaseURL: "https://api.ocoolai.com", DefaultModel: ""},
	providerOllama:            {Name: providerOllama, BaseURL: "http://localhost:11434", DefaultModel: ""},
	"openai":                  {Name: "openai", BaseURL: "https://api.openai.com", DefaultModel: ""},
	"openrouter":              {Name: "openrouter", BaseURL: "https://openrouter.ai/api/v1/", DefaultModel: ""},
	"perplexity":              {Name: "perplexity", BaseURL: "https://api.perplexity.ai/", DefaultModel: ""},
	"ppio":                    {Name: "ppio", BaseURL: "https://api.ppinfra.com/v3/openai", DefaultModel: ""},
	"silicon":                 {Name: "silicon", BaseURL: "https://api.siliconflow.cn", DefaultModel: ""},
	"stepfun":                 {Name: "stepfun", BaseURL: "https://api.stepfun.com", DefaultModel: ""},
	"tencent-cloud-ti": {
		Name: "tencent-cloud-ti", BaseURL: "https://api.lkeap.cloud.tencent.com", DefaultModel: "",
	},
	"together": {Name: "together", BaseURL: "https://api.together.xyz", DefaultModel: ""},
	"xirang":   {Name: "xirang", BaseURL: "https://wishub-x1.ctyun.cn", DefaultModel: ""},
	"yi":       {Name: "yi", BaseURL: "https://api.lingyiwanwu.com", DefaultModel: ""},
	"zhinao":   {Name: "zhinao", BaseURL: "https://api.360.cn", DefaultModel: ""},
	"zhipu":    {Name: "zhipu", BaseURL: "https://open.bigmodel.cn/api/paas/v4/", DefaultModel: ""},
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

func parseModelString(model string) (provider, string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return provider{}, "", fmt.Errorf("model string must not be empty")
	}

	var providerName string
	var modelName string

	if strings.Contains(model, "/") {
		parts := strings.SplitN(model, "/", 2)
		providerName, modelName = parts[0], strings.TrimSpace(parts[1])
	} else {
		providerName = model
	}

	prov, ok := providers[providerName]
	if !ok {
		prov = provider{Name: providerName, BaseURL: "", DefaultModel: ""}
	}

	if modelName == "" {
		if prov.DefaultModel != "" {
			modelName = prov.DefaultModel
		} else {
			return provider{}, "", fmt.Errorf("model name missing for provider %q", providerName)
		}
	}

	return prov, modelName, nil
}
