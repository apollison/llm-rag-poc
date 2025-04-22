package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/andrew/llm-rag-poc/pkg/models"
)

// OllamaClient is a client that uses the Ollama API to interact with LLM models
type OllamaClient struct {
	baseURL    string
	httpClient *http.Client
	modelName  string
}

// OllamaRequest represents a request to the Ollama API
type OllamaRequest struct {
	Model    string    `json:"model"`
	Prompt   string    `json:"prompt,omitempty"`
	Messages []Message `json:"messages,omitempty"`
	Stream   bool      `json:"stream,omitempty"`
	Options  Options   `json:"options,omitempty"`
}

// Message represents a chat message
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Options represents parameter options for the model
type Options struct {
	Temperature float32 `json:"temperature,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	MaxTokens   int     `json:"num_predict,omitempty"`
}

// OllamaResponse represents a response from the Ollama API
type OllamaResponse struct {
	Response string  `json:"response"`
	Model    string  `json:"model"`
	Done     bool    `json:"done"`
	Metrics  Metrics `json:"metrics,omitempty"`
}

// EmbeddingResponse represents a response from the Ollama embeddings API
type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Metrics represents performance metrics from the Ollama API
type Metrics struct {
	PromptEvalCount int     `json:"prompt_eval_count"`
	EvalCount       int     `json:"eval_count"`
	TotalDurationNs int64   `json:"total_duration"`
	LoadDurationNs  int64   `json:"load_duration"`
	PromptEvalSpeed float64 `json:"prompt_eval_speed"`
	EvalSpeed       float64 `json:"eval_speed"`
	TokensPerSecond float64 `json:"tokens_per_second"`
}

// NewOllamaClient creates a new client for interacting with a local Ollama server
func NewOllamaClient(modelName string, baseURL string) *OllamaClient {
	if baseURL == "" {
		baseURL = "http://localhost:11434/api"
	}

	return &OllamaClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: time.Minute * 5, // 5 minute timeout for potentially long generations
		},
		modelName: modelName,
	}
}

// Chat processes a conversation and returns a response
func (c *OllamaClient) Chat(ctx context.Context, messages []models.Message, config ModelConfig) (models.Message, error) {
	// Convert our messages to Ollama format
	ollamaMessages := make([]Message, len(messages))
	for i, msg := range messages {
		ollamaMessages[i] = Message{
			Role:    string(msg.Role),
			Content: msg.Content,
		}
	}

	// Prepare the request
	req := OllamaRequest{
		Model:    c.modelName,
		Messages: ollamaMessages,
		Options: Options{
			Temperature: config.Temperature,
			TopP:        config.TopP,
			MaxTokens:   config.MaxTokens,
		},
	}

	// Send request and handle streaming response
	fullResponse, err := c.sendChatRequest(ctx, req)
	if err != nil {
		return models.Message{}, err
	}

	return models.Message{
		Role:      models.RoleAssistant,
		Content:   fullResponse,
		Timestamp: time.Now(),
	}, nil
}

// sendChatRequest handles the chat endpoint and processes streamed responses
func (c *OllamaClient) sendChatRequest(ctx context.Context, req OllamaRequest) (string, error) {
	// Serialize the request body
	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	// Create the HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/chat", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Send the request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	// Check for error status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Ollama API error (status %d): %s", resp.StatusCode, body)
	}

	// Ollama returns a stream of JSON objects, one per line
	// We need to read them all and concatenate the responses
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()
		// Skip empty lines
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}

		var chunkResponse OllamaResponse
		if err := json.Unmarshal([]byte(line), &chunkResponse); err != nil {
			return "", fmt.Errorf("failed to parse Ollama response chunk: %v", err)
		}

		// Add this chunk's response to our full response
		fullResponse.WriteString(chunkResponse.Response)
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading response stream: %v", err)
	}

	return fullResponse.String(), nil
}

// Generate processes a single prompt and returns a completion
func (c *OllamaClient) Generate(ctx context.Context, prompt string, config ModelConfig) (string, error) {
	// Prepare the request
	req := OllamaRequest{
		Model:  c.modelName,
		Prompt: prompt,
		Options: Options{
			Temperature: config.Temperature,
			TopP:        config.TopP,
			MaxTokens:   config.MaxTokens,
		},
	}

	// Send to the Ollama API and handle streaming response similar to Chat
	fullResponse, err := c.sendGenerateRequest(ctx, req)
	if err != nil {
		return "", err
	}

	return fullResponse, nil
}

// sendGenerateRequest handles the generate endpoint and processes streamed responses
func (c *OllamaClient) sendGenerateRequest(ctx context.Context, req OllamaRequest) (string, error) {
	// Serialize the request body
	reqBody, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %v", err)
	}

	// Create the HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/generate", bytes.NewReader(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Send the request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	// Check for error status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("Ollama API error (status %d): %s", resp.StatusCode, body)
	}

	// Ollama returns a stream of JSON objects, one per line
	// We need to read them all and concatenate the responses
	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()
		// Skip empty lines
		if len(strings.TrimSpace(line)) == 0 {
			continue
		}

		var chunkResponse OllamaResponse
		if err := json.Unmarshal([]byte(line), &chunkResponse); err != nil {
			return "", fmt.Errorf("failed to parse Ollama response chunk: %v", err)
		}

		// Add this chunk's response to our full response
		fullResponse.WriteString(chunkResponse.Response)
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading response stream: %v", err)
	}

	return fullResponse.String(), nil
}

// EmbedText generates vector embeddings for a given text
func (c *OllamaClient) EmbedText(ctx context.Context, text string) ([]float32, error) {
	// Prepare the request
	req := OllamaRequest{
		Model:  c.modelName,
		Prompt: text,
	}

	// Send to the Ollama API
	resp, err := c.sendRequest(ctx, "/embeddings", req)
	if err != nil {
		return nil, err
	}

	// Parse the response
	var embedResp EmbeddingResponse
	if err := json.Unmarshal(resp, &embedResp); err != nil {
		return nil, fmt.Errorf("failed to parse Ollama embeddings response: %v", err)
	}

	return embedResp.Embedding, nil
}

// sendRequest sends a request to the Ollama API - used for non-streaming endpoints
func (c *OllamaClient) sendRequest(ctx context.Context, endpoint string, req interface{}) ([]byte, error) {
	// Serialize the request body
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}

	// Create the HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+endpoint, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// Send the request
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	// Check for error status
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Ollama API error (status %d): %s", resp.StatusCode, body)
	}

	// Read the response
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	return respBody, nil
}

// Close cleans up any resources
func (c *OllamaClient) Close() error {
	// No cleanup needed for HTTP client
	return nil
}
