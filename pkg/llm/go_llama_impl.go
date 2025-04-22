//go:build cgo
// +build cgo

package llm

// This file contains the actual implementation of the LLModel interface
// using our custom binding to llama.cpp.

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/andrew/llm-rag-poc/pkg/llm/binding"
)

// GoLlamaModel is an implementation of LLModel using our custom binding
type GoLlamaModel struct {
	model *binding.LlamaModel
}

// NewGoLlamaModel creates a new GoLlamaModel instance
func NewGoLlamaModel(config LlamaConfig) (*GoLlamaModel, error) {
	// Get absolute path to the model file
	absPath, err := filepath.Abs(config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path to model: %w", err)
	}

	// Create llama model using our binding
	model, err := binding.NewLlamaModel(
		absPath,
		config.ContextSize,
		config.Seed,
		config.Threads,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to initialize llama model: %w", err)
	}

	return &GoLlamaModel{
		model: model,
	}, nil
}

// Predict implements the LLModel interface
func (m *GoLlamaModel) Predict(text string, options ...PredictOption) (string, error) {
	if m.model == nil {
		return "", errors.New("model not initialized")
	}

	// Apply options
	opts := make(map[string]interface{})
	for _, option := range options {
		option(opts)
	}

	// Extract options with default values
	temp := float32(0.7)
	if tempVal, ok := opts["temperature"].(float32); ok {
		temp = tempVal
	}

	topP := float32(0.95)
	if topPVal, ok := opts["top_p"].(float32); ok {
		topP = topPVal
	}

	tokens := 2048
	if tokensVal, ok := opts["tokens"].(int); ok {
		tokens = tokensVal
	}

	seed := 42
	topK := 40
	repeatPenalty := float32(1.1)

	// Call the model's Predict method
	return m.model.Predict(
		text,
		tokens,
		seed,
		0, // use model's default thread count
		temp,
		topP,
		topK,
		repeatPenalty,
	)
}

// Embeddings implements the LLModel interface
func (m *GoLlamaModel) Embeddings(text string) ([]float32, error) {
	// Our simplified binding doesn't support embeddings yet
	return nil, errors.New("embeddings not currently supported by the simplified binding")
}

// Close implements the LLModel interface
func (m *GoLlamaModel) Close() error {
	if m.model == nil {
		return nil
	}
	m.model.Free()
	return nil
}

// CreateLlamaClient creates a LlamaClient with our GoLlamaModel implementation
func CreateLlamaClient(config LlamaConfig) (*LlamaClient, error) {
	model, err := NewGoLlamaModel(config)
	if err != nil {
		return nil, err
	}

	return &LlamaClient{
		model:  model,
		config: config,
	}, nil
}
