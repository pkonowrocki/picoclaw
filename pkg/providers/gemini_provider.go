package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sipeed/picoclaw/pkg/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/genai"
)

type GeminiProvider struct {
	apiKey         string
	apiBase        string
	keyRotator     *KeyRotator
	fallbackModels []string
	client         *genai.Client

	lastModel string
	lastKey   string
	mu        sync.RWMutex
}

var geminiFallbackModels = []string{
	"gemini-2.5-pro",
	"gemini-flash-latest",
	"gemini-3-pro-preview",
	"gemini-2.5-flash-lite",
}

func NewGeminiProvider(apiKey, apiBase, proxy string) *GeminiProvider {
	apiBase = strings.TrimRight(apiBase, "/")
	apiBase = strings.TrimSuffix(apiBase, "/openai")

	if apiBase == "" {
		apiBase = "https://generativelanguage.googleapis.com"
	}

	return &GeminiProvider{
		apiKey:  apiKey,
		apiBase: apiBase,
	}
}

func NewGeminiProviderWithKeys(keys []string, apiBase, proxy string) *GeminiProvider {
	p := NewGeminiProvider(keys[0], apiBase, proxy)
	if p != nil {
		p.keyRotator = NewKeyRotator(keys)
	}
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

	if idx := strings.Index(model, "/"); idx != -1 {
		prefix := model[:idx]
		if prefix == "google" || prefix == "gemini" {
			model = model[idx+1:]
		}
	}

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

	keys := []string{p.apiKey}
	if p.keyRotator != nil {
		keys = p.keyRotator.GetAll()
	}

	p.mu.RLock()
	cachedModel, cachedKey := p.lastModel, p.lastKey
	p.mu.RUnlock()

	if cachedModel != "" && cachedKey != "" {
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
			if currentModel == cachedModel && apiKey == cachedKey {
				continue
			}

			response, err := p.tryRequest(ctx, messages, tools, currentModel, apiKey, options)
			if err == nil {
				p.mu.Lock()
				p.lastModel = currentModel
				p.lastKey = apiKey
				p.mu.Unlock()
				return response, nil
			}

			lastErr = err

			if strings.Contains(err.Error(), "RESOURCE_EXHAUSTED") ||
				strings.Contains(err.Error(), "quota") ||
				strings.Contains(err.Error(), "Quota") {
				continue
			}
			break
		}
	}

	return nil, fmt.Errorf("all Gemini models and keys exhausted: %w", lastErr)
}

func (p *GeminiProvider) tryRequest(ctx context.Context, messages []Message, tools []ToolDefinition, model, apiKey string, options map[string]interface{}) (*LLMResponse, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
		HTTPClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	contents := p.buildContents(messages)

	genConfig := &genai.GenerateContentConfig{}
	if len(tools) > 0 {
		genTools := make([]*genai.Tool, len(tools))
		for i, t := range tools {
			genTools[i] = &genai.Tool{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        t.Function.Name,
						Description: t.Function.Description,
						Parameters:  convertParameters(t.Function.Parameters),
					},
				},
			}
		}
		genConfig.Tools = genTools
		genConfig.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: genai.FunctionCallingConfigModeAuto,
			},
		}
	}

	if maxTokens, ok := options["max_tokens"].(int); ok {
		genConfig.MaxOutputTokens = int32(maxTokens)
	}
	if temperature, ok := options["temperature"].(float64); ok {
		genConfig.Temperature = genai.Ptr(float32(temperature))
	}

	resp, err := client.Models.GenerateContent(ctx, model, contents, genConfig)
	if err != nil {
		return nil, fmt.Errorf("Gemini API error: %w", err)
	}

	return p.parseResponse(resp)
}

func convertParameters(params map[string]interface{}) *genai.Schema {
	if params == nil {
		return nil
	}
	return &genai.Schema{
		Type: genai.TypeObject,
	}
}

func (p *GeminiProvider) buildContents(messages []Message) []*genai.Content {
	var contents []*genai.Content

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			contents = append(contents, &genai.Content{
				Role: "system",
				Parts: []*genai.Part{
					{Text: msg.Content},
				},
			})

		case "user":
			parts := p.userParts(msg)
			contents = append(contents, &genai.Content{
				Role:  "user",
				Parts: parts,
			})

		case "assistant":
			parts := p.assistantParts(msg)
			contents = append(contents, &genai.Content{
				Role:  "model",
				Parts: parts,
			})

		case "tool":
			funcResponses := p.toolParts(msg)
			contents = append(contents, &genai.Content{
				Role:  "user",
				Parts: funcResponses,
			})
		}
	}

	return contents
}

func (p *GeminiProvider) userParts(msg Message) []*genai.Part {
	var parts []*genai.Part

	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
	}

	for _, cp := range msg.ContentParts {
		switch cp.Type {
		case "text":
			parts = append(parts, &genai.Part{Text: cp.Text})
		case "image_url":
			if cp.ImageURL != nil {
				if part := geminiPartFromImageURL(cp.ImageURL.URL); part != nil {
					parts = append(parts, part)
				}
			}
		}
	}

	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: ""})
	}
	return parts
}

func (p *GeminiProvider) assistantParts(msg Message) []*genai.Part {
	var parts []*genai.Part

	if msg.Content != "" {
		parts = append(parts, &genai.Part{Text: msg.Content})
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

		var thoughtSig []byte
		if tc.ExtraContent != nil {
			if sig, ok := tc.ExtraContent["thought_signature"].(string); ok && sig != "" {
				thoughtSig = []byte(sig)
			}
		}

		part := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				Name: name,
				Args: args,
			},
			ThoughtSignature: thoughtSig,
		}
		parts = append(parts, part)
	}

	if len(parts) == 0 {
		parts = append(parts, &genai.Part{Text: ""})
	}
	return parts
}

func (p *GeminiProvider) toolParts(msg Message) []*genai.Part {
	funcName := msg.ToolCallID
	if funcName == "" {
		funcName = "unknown"
	}

	var responseData interface{}
	if err := json.Unmarshal([]byte(msg.Content), &responseData); err != nil {
		responseData = map[string]interface{}{"result": msg.Content}
	}
	switch responseData.(type) {
	case []interface{}:
		responseData = map[string]interface{}{"result": responseData}
	}

	return []*genai.Part{
		{
			FunctionResponse: &genai.FunctionResponse{
				Name: funcName,
				Response: map[string]interface{}{
					"result": responseData,
				},
			},
		},
	}
}

func geminiPartFromImageURL(imageURL string) *genai.Part {
	if strings.HasPrefix(imageURL, "data:") {
		rest := imageURL[5:]
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

		return &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: mimeType,
				Data:     []byte(base64Data),
			},
		}
	}

	if strings.HasPrefix(imageURL, "https://") || strings.HasPrefix(imageURL, "http://") {
		return &genai.Part{
			FileData: &genai.FileData{
				FileURI: imageURL,
			},
		}
	}

	return nil
}

func (p *GeminiProvider) parseResponse(resp *genai.GenerateContentResponse) (*LLMResponse, error) {
	if len(resp.Candidates) == 0 {
		return &LLMResponse{Content: "", FinishReason: "stop"}, nil
	}

	candidate := resp.Candidates[0]
	var content string
	var toolCalls []ToolCall

	for i, part := range candidate.Content.Parts {
		if part.Text != "" {
			content += part.Text
		}
		if part.FunctionCall != nil {
			if len(part.ThoughtSignature) == 0 {
				log.Printf("[GEMINI] WARNING: Function call %d (%s) has no thought_signature", i, part.FunctionCall.Name)
			}
			tc := ToolCall{
				ID:        fmt.Sprintf("call_%s", uuid.New().String()[:8]),
				Name:      part.FunctionCall.Name,
				Arguments: part.FunctionCall.Args,
				ExtraContent: map[string]interface{}{
					"thought_signature": string(part.ThoughtSignature),
				},
			}
			toolCalls = append(toolCalls, tc)
		}
	}

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	usage := &UsageInfo{}
	if resp.UsageMetadata != nil {
		usage.PromptTokens = int(resp.UsageMetadata.PromptTokenCount)
		usage.CompletionTokens = int(resp.UsageMetadata.CandidatesTokenCount)
		usage.TotalTokens = int(resp.UsageMetadata.TotalTokenCount)
	}

	return &LLMResponse{
		Content:      content,
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}
