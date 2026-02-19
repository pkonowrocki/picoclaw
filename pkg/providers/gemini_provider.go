package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sipeed/picoclaw/pkg/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type GeminiProvider struct {
	apiKey         string
	apiBase        string
	httpClient     *http.Client
	keyRotator     *KeyRotator
	fallbackModels []string

	// Cache for last successful model+key
	lastModel string
	lastKey   string
	mu        sync.RWMutex
}

var geminiFallbackModels = []string{
	"gemini-2.5-pro",
	"gemini-2.5-flash",
	"gemini-2.5-flash-lite-preview-06-17",
	// "gemini-2.0-flash",
	// "gemini-2.0-flash-lite",
}

func NewGeminiProvider(apiKey, apiBase, proxy string) *GeminiProvider {
	client := &http.Client{
		Timeout: 120 * time.Second,
	}

	if proxy != "" {
		proxyURL, err := url.Parse(proxy)
		if err == nil {
			client.Transport = &http.Transport{
				Proxy: http.ProxyURL(proxyURL),
			}
		}
	}

	// Normalize API base: strip /openai suffix if present (migration from old config)
	apiBase = strings.TrimRight(apiBase, "/")
	apiBase = strings.TrimSuffix(apiBase, "/openai")

	if apiBase == "" {
		apiBase = "https://generativelanguage.googleapis.com/v1beta"
	}

	return &GeminiProvider{
		apiKey:     apiKey,
		apiBase:    apiBase,
		httpClient: client,
	}
}

func NewGeminiProviderWithKeys(keys []string, apiBase, proxy string) *GeminiProvider {
	p := NewGeminiProvider(keys[0], apiBase, proxy)
	p.keyRotator = NewKeyRotator(keys)
	return p
}

func (p *GeminiProvider) GetDefaultModel() string {
	return "gemini-3-flash-preview"
}

func (p *GeminiProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]interface{}) (*LLMResponse, error) {
	ctx, span := tracing.Tracer("provider").Start(ctx, "provider.gemini.chat",
		trace.WithAttributes(
			attribute.String("model", model),
			attribute.Int("messages_count", len(messages)),
		))
	defer span.End()

	// Strip provider prefix from model name (e.g. google/gemini-2.0-flash → gemini-2.0-flash)
	if idx := strings.Index(model, "/"); idx != -1 {
		prefix := model[:idx]
		if prefix == "google" || prefix == "gemini" {
			model = model[idx+1:]
		}
	}

	// Build model list: primary first, then fallbacks
	models := []string{model}
	for _, fb := range p.fallbackModels {
		if fb != model {
			models = append(models, fb)
		}
	}
	if len(models) == 1 && len(geminiFallbackModels) > 0 {
		for _, fb := range geminiFallbackModels {
			if fb != model {
				models = append(models, fb)
			}
		}
	}

	// Build key list
	keys := []string{p.apiKey}
	if p.keyRotator != nil {
		keys = p.keyRotator.GetAll()
	}

	// Check cache for last successful combo and try it first
	p.mu.RLock()
	cachedModel, cachedKey := p.lastModel, p.lastKey
	p.mu.RUnlock()

	if cachedModel != "" && cachedKey != "" {
		// Verify cached values are still valid
		modelValid := false
		for _, m := range models {
			if m == cachedModel {
				modelValid = true
				break
			}
		}
		keyValid := false
		for _, k := range keys {
			if k == cachedKey {
				keyValid = true
				break
			}
		}

		if modelValid && keyValid {
			response, err := p.tryRequest(ctx, messages, tools, cachedModel, cachedKey, options)
			if err == nil {
				return response, nil
			}
		}
	}

	var lastErr error
	for _, currentModel := range models {
		for _, apiKey := range keys {
			// Skip cached combo (already tried)
			if currentModel == cachedModel && apiKey == cachedKey {
				continue
			}

			response, err := p.tryRequest(ctx, messages, tools, currentModel, apiKey, options)
			if err == nil {
				// Cache successful combo
				p.mu.Lock()
				p.lastModel = currentModel
				p.lastKey = apiKey
				p.mu.Unlock()
				return response, nil
			}

			lastErr = err

			// Quota exhaustion - try next key
			if strings.Contains(err.Error(), "RESOURCE_EXHAUSTED") ||
				strings.Contains(err.Error(), "quota") ||
				strings.Contains(err.Error(), "Quota") {
				continue
			}

			// Other errors - try next model
			break
		}
	}

	return nil, fmt.Errorf("all Gemini models and keys exhausted: %w", lastErr)
}

func (p *GeminiProvider) tryRequest(ctx context.Context, messages []Message, tools []ToolDefinition, model, apiKey string, options map[string]interface{}) (*LLMResponse, error) {
	requestBody := p.buildRequest(messages, tools, options)

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Gemini request: %w", err)
	}

	endpoint := fmt.Sprintf("%s/models/%s:generateContent", p.apiBase, model)
	reqURL := fmt.Sprintf("%s?key=%s", endpoint, apiKey)

	req, err := http.NewRequestWithContext(ctx, "POST", reqURL, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}

	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode == http.StatusOK {
		return p.parseResponse(body)
	}

	return nil, fmt.Errorf("Gemini API error (HTTP %d): %s", resp.StatusCode, string(body))
}

// buildRequest converts the common message format to a native Gemini API request.
func (p *GeminiProvider) buildRequest(messages []Message, tools []ToolDefinition, options map[string]interface{}) map[string]interface{} {
	request := make(map[string]interface{})

	var systemParts []map[string]interface{}
	var contents []map[string]interface{}

	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		switch msg.Role {
		case "system":
			systemParts = append(systemParts, map[string]interface{}{
				"text": msg.Content,
			})

		case "user":
			parts := p.userParts(msg)
			contents = append(contents, map[string]interface{}{
				"role":  "user",
				"parts": parts,
			})

		case "assistant":
			parts := p.assistantParts(msg)
			contents = append(contents, map[string]interface{}{
				"role":  "model",
				"parts": parts,
			})

		case "tool":
			// Group consecutive tool messages into a single user turn with functionResponse parts
			var funcResponses []map[string]interface{}
			for ; i < len(messages) && messages[i].Role == "tool"; i++ {
				toolMsg := messages[i]
				funcName := p.resolveToolName(messages[:i], toolMsg.ToolCallID)

				var responseData interface{}
				if err := json.Unmarshal([]byte(toolMsg.Content), &responseData); err != nil {
					responseData = map[string]interface{}{"result": toolMsg.Content}
				}
				switch responseData.(type) {
				case []interface{}:
					responseData = map[string]interface{}{"result": responseData}
				}

				funcResponses = append(funcResponses, map[string]interface{}{
					"functionResponse": map[string]interface{}{
						"name":     funcName,
						"response": responseData,
					},
				})
			}
			i-- // compensate for outer loop increment

			contents = append(contents, map[string]interface{}{
				"role":  "user",
				"parts": funcResponses,
			})
		}
	}

	if len(systemParts) > 0 {
		request["system_instruction"] = map[string]interface{}{
			"parts": systemParts,
		}
	}
	if len(contents) > 0 {
		request["contents"] = contents
	}

	// Tool definitions
	if len(tools) > 0 {
		funcDecls := make([]map[string]interface{}, 0, len(tools))
		for _, t := range tools {
			decl := map[string]interface{}{
				"name":        t.Function.Name,
				"description": t.Function.Description,
			}
			if t.Function.Parameters != nil {
				decl["parameters"] = t.Function.Parameters
			}
			funcDecls = append(funcDecls, decl)
		}
		request["tools"] = []map[string]interface{}{
			{"function_declarations": funcDecls},
		}
		request["tool_config"] = map[string]interface{}{
			"function_calling_config": map[string]interface{}{
				"mode": "AUTO",
			},
		}
	}

	// Generation config
	genConfig := map[string]interface{}{}
	if maxTokens, ok := options["max_tokens"].(int); ok {
		genConfig["maxOutputTokens"] = maxTokens
	}
	if temperature, ok := options["temperature"].(float64); ok {
		genConfig["temperature"] = temperature
	}
	if len(genConfig) > 0 {
		request["generationConfig"] = genConfig
	}

	return request
}

func (p *GeminiProvider) userParts(msg Message) []map[string]interface{} {
	if len(msg.ContentParts) > 0 {
		parts := make([]map[string]interface{}, 0, len(msg.ContentParts))
		for _, cp := range msg.ContentParts {
			switch cp.Type {
			case "text":
				parts = append(parts, map[string]interface{}{"text": cp.Text})
			case "image_url":
				if cp.ImageURL != nil {
					if imgPart := geminiImagePart(cp.ImageURL.URL); imgPart != nil {
						parts = append(parts, imgPart)
					}
				}
			}
		}
		if len(parts) == 0 {
			parts = append(parts, map[string]interface{}{"text": ""})
		}
		return parts
	}
	return []map[string]interface{}{{"text": msg.Content}}
}

func (p *GeminiProvider) assistantParts(msg Message) []map[string]interface{} {
	var parts []map[string]interface{}

	if msg.Content != "" {
		parts = append(parts, map[string]interface{}{"text": msg.Content})
	}

	for _, tc := range msg.ToolCalls {
		name := tc.Name
		if name == "" && tc.Function != nil {
			name = tc.Function.Name
		}
		args := tc.Arguments
		if args == nil && tc.Function != nil && tc.Function.Arguments != "" {
			_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
		}
		if args == nil {
			args = map[string]interface{}{}
		}
		fcPart := map[string]interface{}{
			"name": name,
			"args": args,
		}
		part := map[string]interface{}{
			"functionCall": fcPart,
		}
		if sig, ok := tc.ExtraContent["thought_signature"].(string); ok && sig != "" {
			part["thoughtSignature"] = sig
		}
		parts = append(parts, part)
	}

	if len(parts) == 0 {
		parts = append(parts, map[string]interface{}{"text": ""})
	}
	return parts
}

// geminiImagePart converts an image URL to a Gemini-native part.
//   - data: URIs  → inlineData  (base64 images from MediaToDataURI)
//   - https: URLs → fileData    (remote images Gemini can fetch)
func geminiImagePart(imageURL string) map[string]interface{} {
	if strings.HasPrefix(imageURL, "data:") {
		// Parse data:{mimeType};base64,{data}
		rest := imageURL[5:] // strip "data:"
		semiIdx := strings.Index(rest, ";")
		if semiIdx == -1 {
			return nil
		}
		mimeType := rest[:semiIdx]

		const marker = ";base64,"
		dataStart := strings.Index(imageURL, marker)
		if dataStart == -1 {
			return nil
		}
		base64Data := imageURL[dataStart+len(marker):]

		return map[string]interface{}{
			"inlineData": map[string]interface{}{
				"mimeType": mimeType,
				"data":     base64Data,
			},
		}
	}

	if strings.HasPrefix(imageURL, "https://") || strings.HasPrefix(imageURL, "http://") {
		return map[string]interface{}{
			"fileData": map[string]interface{}{
				"fileUri":  imageURL,
				"mimeType": "image/jpeg",
			},
		}
	}

	return nil
}

// resolveToolName finds the function name for a tool_call_id by searching
// previous assistant messages (Gemini matches tool results by name, not ID).
func (p *GeminiProvider) resolveToolName(preceding []Message, toolCallID string) string {
	for i := len(preceding) - 1; i >= 0; i-- {
		if preceding[i].Role == "assistant" {
			for _, tc := range preceding[i].ToolCalls {
				if tc.ID == toolCallID {
					if tc.Name != "" {
						return tc.Name
					}
					if tc.Function != nil {
						return tc.Function.Name
					}
				}
			}
		}
	}
	return "unknown"
}

func (p *GeminiProvider) parseResponse(body []byte) (*LLMResponse, error) {
	var resp struct {
		Candidates []struct {
			Content struct {
				Parts []json.RawMessage `json:"parts"`
				Role  string            `json:"role"`
			} `json:"content"`
			FinishReason string `json:"finishReason"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}

	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("failed to parse Gemini response: %w (body: %.200s)", err, string(body))
	}

	if len(resp.Candidates) == 0 {
		return &LLMResponse{Content: "", FinishReason: "stop"}, nil
	}

	candidate := resp.Candidates[0]
	var content string
	var toolCalls []ToolCall

	for _, raw := range candidate.Content.Parts {
		var part map[string]interface{}
		if err := json.Unmarshal(raw, &part); err != nil {
			continue
		}

		if text, ok := part["text"].(string); ok {
			content += text
		}

		if fc, ok := part["functionCall"].(map[string]interface{}); ok {
			name, _ := fc["name"].(string)
			args, _ := fc["args"].(map[string]interface{})
			if args == nil {
				args = map[string]interface{}{}
			}
			tc := ToolCall{
				ID:        fmt.Sprintf("call_%s", uuid.New().String()[:8]),
				Name:      name,
				Arguments: args,
			}
			if sig, ok := part["thoughtSignature"].(string); ok && sig != "" {
				tc.ExtraContent = map[string]interface{}{
					"thought_signature": sig,
				}
			}
			toolCalls = append(toolCalls, tc)
		}
	}

	finishReason := "stop"
	switch candidate.FinishReason {
	case "MAX_TOKENS":
		finishReason = "length"
	}
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	return &LLMResponse{
		Content:      content,
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage: &UsageInfo{
			PromptTokens:     resp.UsageMetadata.PromptTokenCount,
			CompletionTokens: resp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      resp.UsageMetadata.TotalTokenCount,
		},
	}, nil
}
