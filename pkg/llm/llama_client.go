package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/andrew/llm-rag-poc/pkg/models"
)

// LlamaConfig holds configuration for the Llama model
type LlamaConfig struct {
	ModelPath          string
	ContextSize        int
	Threads            int
	Seed               int
	F16Memory          bool
	VocabOnly          bool
	UseMMap            bool
	UseMlock           bool
	UseNGL             int    // Number of GPU layers, 0 for CPU only, >0 for GPU
	UseMetalShaderPath string // Path to the Metal shader for Mac
}

// DefaultLlamaConfig returns a default configuration for Llama
func DefaultLlamaConfig() LlamaConfig {
	return LlamaConfig{
		ModelPath:          "models/llama3.2.gguf",
		ContextSize:        2048,
		Threads:            4,
		Seed:               42,
		F16Memory:          true,
		UseMMap:            true,
		UseMlock:           false,
		UseNGL:             1,                    // Default to using 1 GPU layer if available
		UseMetalShaderPath: "./ggml-metal.metal", // Default Metal shader path for Mac
	}
}

// LlamaClient is a client that uses go-llama.cpp to interact with Llama models
type LlamaClient struct {
	model  LLModel
	config LlamaConfig
}

// LLModel is an interface that wraps the necessary methods for interacting with llama.cpp
// This allows us to mock it for testing or switch implementations if needed
type LLModel interface {
	Predict(text string, options ...PredictOption) (string, error)
	Embeddings(text string) ([]float32, error)
	Close() error
}

// PredictOption is a function that configures prediction options
type PredictOption func(map[string]interface{})

// WithTemperature sets the temperature for sampling
func WithTemperature(temp float32) PredictOption {
	return func(opts map[string]interface{}) {
		opts["temperature"] = temp
	}
}

// WithTopP sets the top_p for sampling
func WithTopP(topP float32) PredictOption {
	return func(opts map[string]interface{}) {
		opts["top_p"] = topP
	}
}

// WithTokens sets the max number of tokens to generate
func WithTokens(tokens int) PredictOption {
	return func(opts map[string]interface{}) {
		opts["tokens"] = tokens
	}
}

// WithStop sets stop sequences for generation
func WithStop(stop []string) PredictOption {
	return func(opts map[string]interface{}) {
		opts["stop"] = stop
	}
}

// NewLlamaClient creates a new Llama client using our locally built llama.cpp bindings
func NewLlamaClient(config LlamaConfig) (*LlamaClient, error) {
	// When building with CGO, this function is overridden by the implementation in go_llama_impl.go
	// For non-CGO builds, return an error
	return nil, fmt.Errorf("go-llama.cpp bindings not properly set up. Build with -tags cgo")
}

// formatMessages formats messages for the Llama model
func formatMessages(messages []models.Message) string {
	var prompt string

	for _, msg := range messages {
		switch msg.Role {
		case models.RoleSystem:
			prompt += fmt.Sprintf("<|system|>\n%s\n", msg.Content)
		case models.RoleUser:
			prompt += fmt.Sprintf("<|user|>\n%s\n", msg.Content)
		case models.RoleAssistant:
			prompt += fmt.Sprintf("<|assistant|>\n%s\n", msg.Content)
		}
	}

	// Add the final assistant prefix to indicate where the model should continue
	prompt += "<|assistant|>\n"

	return prompt
}

// Chat processes a conversation and returns a response
func (c *LlamaClient) Chat(ctx context.Context, messages []models.Message, config ModelConfig) (models.Message, error) {
	prompt := formatMessages(messages)

	// Configure model parameters as PredictOptions
	options := []PredictOption{
		WithTemperature(config.Temperature),
		WithTopP(config.TopP),
		WithTokens(config.MaxTokens),
		WithStop(config.StopSequences),
	}

	// Generate response
	output, err := c.model.Predict(prompt, options...)
	if err != nil {
		return models.Message{}, fmt.Errorf("failed to generate response: %v", err)
	}

	// Create and return the assistant's message
	return models.Message{
		Role:      models.RoleAssistant,
		Content:   output,
		Timestamp: time.Now(),
	}, nil
}

// Generate processes a single prompt and returns a completion
func (c *LlamaClient) Generate(ctx context.Context, prompt string, config ModelConfig) (string, error) {
	// Configure model parameters as PredictOptions
	options := []PredictOption{
		WithTemperature(config.Temperature),
		WithTopP(config.TopP),
		WithTokens(config.MaxTokens),
		WithStop(config.StopSequences),
	}

	// Generate response
	output, err := c.model.Predict(prompt, options...)
	if err != nil {
		return "", fmt.Errorf("failed to generate response: %v", err)
	}

	return output, nil
}

// EmbedText generates vector embeddings for a given text
func (c *LlamaClient) EmbedText(ctx context.Context, text string) ([]float32, error) {
	// Generate embeddings
	embeddings, err := c.model.Embeddings(text)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embeddings: %v", err)
	}

	return embeddings, nil
}

// Close releases resources used by the model
func (c *LlamaClient) Close() error {
	if c.model != nil {
		return c.model.Close()
	}
	return nil
}
