package smolllm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAskResolvesBaseURLByPrecedence(t *testing.T) {
	// Four tiers, strongest first: WithProvider, WithDefaultProvider, the
	// ${PROVIDER}_BASE_URL environment variable, then the provider table.
	tests := []struct {
		name        string
		model       string
		envKey      string
		envURL      string
		providerURL string
		defaultURL  string
		wantURL     string
	}{
		{
			name:        "WithProvider beats every weaker tier",
			model:       "openai/gpt-5",
			envKey:      "OPENAI_BASE_URL",
			envURL:      "https://env.example/v2/",
			providerURL: "https://provider.example/api/",
			defaultURL:  "https://default.example/api/",
			wantURL:     "https://provider.example/api/chat/completions",
		},
		{
			name:        "WithDefaultProvider beats environment and provider table",
			model:       "openai/gpt-5",
			envKey:      "OPENAI_BASE_URL",
			envURL:      "https://env.example/v2/",
			providerURL: "",
			defaultURL:  "https://default.example/api/",
			wantURL:     "https://default.example/api/chat/completions",
		},
		{
			name:        "environment overrides provider table",
			model:       "openai/gpt-5",
			envKey:      "OPENAI_BASE_URL",
			envURL:      "https://env.example/v2/",
			providerURL: "",
			defaultURL:  "",
			wantURL:     "https://env.example/v2/chat/completions",
		},
		{
			name:        "provider table is the fallback",
			model:       "openai/gpt-5",
			envKey:      "OPENAI_BASE_URL",
			envURL:      "",
			providerURL: "",
			defaultURL:  "",
			wantURL:     "https://api.openai.com/v1/chat/completions",
		},
		{
			name:        "WithProvider rescues an unknown provider",
			model:       "custom/model-x",
			envKey:      "CUSTOM_BASE_URL",
			envURL:      "",
			providerURL: "https://custom.example/v1/",
			defaultURL:  "",
			wantURL:     "https://custom.example/v1/chat/completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv(tt.envKey, tt.envURL)

			var actualURL string
			client := &http.Client{
				Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
					actualURL = req.URL.String()
					return testHTTPResponse(
						req,
						http.StatusOK,
						"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n",
					), nil
				}),
				CheckRedirect: nil,
				Jar:           nil,
				Timeout:       0,
			}

			options := []Option{
				WithModel(tt.model),
				WithDefaultProvider(testProviderConfig(tt.defaultURL, "test-key")),
				WithHTTPClient(client),
			}
			if tt.providerURL != "" {
				prov, _, err := parseModelString(tt.model)
				require.NoError(t, err)
				options = append(options,
					WithProvider(prov.Name, testProviderConfig(tt.providerURL, "")))
			}

			msg := Ask(context.Background(), RequestFromString("hi"), options...)
			requireAnswered(t, msg)
			assert.Equal(t, "hello", msg.Content)
			assert.Equal(t, tt.wantURL, actualURL)
		})
	}
}

func TestAskAppliesProviderHeaders(t *testing.T) {
	t.Parallel()

	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Clone()
		writeChatSuccess(t, w, "hello", "stop")
	}))
	defer srv.Close()

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("openai/gpt-5"),
		WithDefaultProvider(ProviderConfig{
			BaseURL: srv.URL + "/",
			APIKey:  "default-key",
			Headers: map[string]string{"X-Shared": "from-default", "X-Only-Default": "kept"},
		}),
		WithProvider("openai", ProviderConfig{
			BaseURL: "",
			APIKey:  "",
			// A provider entry wins per key and may override a standard header.
			Headers: map[string]string{"X-Shared": "from-provider", "Authorization": "Bearer override"},
		}),
	)
	requireAnswered(t, msg)

	assert.Equal(t, "from-provider", got.Get("X-Shared"))
	assert.Equal(t, "kept", got.Get("X-Only-Default"))
	assert.Equal(t, "Bearer override", got.Get("Authorization"))
	assert.Equal(t, "application/json", got.Get("Content-Type"))
}

func TestAskBareModelResolvesExplicitOptions(t *testing.T) {
	t.Parallel()

	var actualURL string
	var captured map[string]any
	client := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			actualURL = req.URL.String()
			if err := json.NewDecoder(req.Body).Decode(&captured); err != nil {
				return nil, err
			}
			return testHTTPResponse(
				req,
				http.StatusOK,
				"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n",
			), nil
		}),
		CheckRedirect: nil,
		Jar:           nil,
		Timeout:       0,
	}

	var hooked []Attempt
	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("gpt-4"),
		WithReasoningEffort("low"),
		WithProvider(BareProvider, testProviderConfig("https://bare.example", "test-key")),
		WithHTTPClient(client),
		WithHook(func(attempt Attempt) {
			hooked = append(hooked, attempt)
		}),
	)
	requireAnswered(t, msg)

	assert.Equal(t, "https://bare.example/v1/chat/completions", actualURL)
	assert.Equal(t, "gpt-4", captured["model"])
	assert.Equal(t, "low", captured["reasoning_effort"])
	assert.Equal(t, "hello", msg.Content)
	assert.Equal(t, "gpt-4", msg.Model)
	assert.Equal(t, "gpt-4", msg.ModelName)
	assert.Empty(t, msg.Provider)
	require.Len(t, hooked, 1)
	assert.Empty(t, hooked[0].Provider)
	assert.Equal(t, "gpt-4", hooked[0].Model)
}

func TestAskChainMixesBareAndPrefixedLegs(t *testing.T) {
	// No base URL configured for the bare leg: it fails on its own config and the
	// chain falls through to the prefixed leg, which resolves from the table.
	t.Setenv("OPENAI_BASE_URL", "")

	var actualURL string
	client := &http.Client{
		Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			actualURL = req.URL.String()
			return testHTTPResponse(
				req,
				http.StatusOK,
				"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n",
			), nil
		}),
		CheckRedirect: nil,
		Jar:           nil,
		Timeout:       0,
	}

	msg := Ask(context.Background(), RequestFromString("hi"),
		WithModel("bare-model,openai/gpt-5"),
		withTestProvider("", "test-key"),
		WithHTTPClient(client),
	)
	requireAnswered(t, msg)

	assert.Equal(t, "https://api.openai.com/v1/chat/completions", actualURL)
	assert.Equal(t, "hello", msg.Content)
	assert.Equal(t, "openai", msg.Provider)
	assert.Equal(t, "openai/gpt-5", msg.Model)
}
