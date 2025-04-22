package llm

import (
	"context"
	"os"

	"github.com/andrew/llm-rag-poc/pkg/models"
)

// Client is the interface for interacting with LLMs
type Client interface {
	Chat(ctx context.Context, messages []models.Message, config ModelConfig) (models.Message, error)
	Generate(ctx context.Context, prompt string, config ModelConfig) (string, error)
	EmbedText(ctx context.Context, text string) ([]float32, error)
	Close() error
}

// ModelConfig holds configuration parameters for model generation
type ModelConfig struct {
	Temperature   float32
	TopP          float32
	MaxTokens     int
	StopSequences []string
}

// DefaultModelConfig returns a default configuration
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		Temperature: 0.7,
		TopP:        0.9,
		MaxTokens:   2048,
	}
}

// NewClient creates a new LLM client, defaulting to Ollama
func NewClient() (Client, error) {
	// Check if OLLAMA_MODEL env var is set, otherwise use a default
	modelName := os.Getenv("OLLAMA_MODEL")
	if modelName == "" {
		modelName = "llama3" // Default to llama3 if not specified
	}

	// Check if OLLAMA_API_URL env var is set, otherwise use default
	ollamaURL := os.Getenv("OLLAMA_API_URL")

	// Return Ollama client
	return NewOllamaClient(modelName, ollamaURL), nil
}
