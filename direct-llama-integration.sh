#!/bin/bash
set -e

echo "Building direct llama.cpp integration"

# Set up directories
ROOT_DIR=$(pwd)
LLAMA_CPP_DIR="$ROOT_DIR/vendor/go-llama.cpp/llama.cpp"
INTEGRATION_DIR="$ROOT_DIR/pkg/llm/llama_integration"
mkdir -p "$INTEGRATION_DIR"

# Ensure llama.cpp is built
if [ ! -f "$LLAMA_CPP_DIR/build/libllama.a" ]; then
  echo "Building llama.cpp..."
  cd "$LLAMA_CPP_DIR"
  mkdir -p build
  cd build
  cmake -DBUILD_SHARED_LIBS=OFF ..
  make -j4
  cd "$ROOT_DIR"
else
  echo "llama.cpp already built"
fi

# Create direct C binding
cat > "$INTEGRATION_DIR/binding.h" << 'EOF'
#ifndef LLAMA_BINDING_H
#define LLAMA_BINDING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Handle types for opaque pointers
typedef void* llama_model_handle;
typedef void* llama_context_handle;

// Initialize llama.cpp
void llama_binding_initialize();

// Load a model from a file
llama_model_handle llama_binding_load_model(const char* path, int n_ctx, uint32_t seed, int n_threads);

// Create a context from a model
llama_context_handle llama_binding_create_context(llama_model_handle model);

// Generate text
const char* llama_binding_generate(llama_context_handle ctx, const char* prompt,
                                  int max_tokens, float temperature, float top_p, int top_k);

// Free a context
void llama_binding_free_context(llama_context_handle ctx);

// Free a model
void llama_binding_free_model(llama_model_handle model);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_BINDING_H
EOF

# Create C++ implementation
cat > "$INTEGRATION_DIR/binding.cpp" << 'EOF'
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <thread>
#include <iostream>
#include <memory>

#include "llama.h"
#include "binding.h"

// Result storage for text generation
thread_local std::string generation_result;

void llama_binding_initialize() {
    // Initialize llama backend
    llama_backend_init(false);
}

llama_model_handle llama_binding_load_model(const char* path, int n_ctx, uint32_t seed, int n_threads) {
    // Set up model parameters
    llama_model_params mparams = llama_model_default_params();

    // Load the model
    llama_model* model = llama_load_model_from_file(path, mparams);
    if (!model) {
        std::cerr << "Failed to load model from " << path << std::endl;
        return nullptr;
    }

    return static_cast<llama_model_handle>(model);
}

llama_context_handle llama_binding_create_context(llama_model_handle model_handle) {
    if (!model_handle) {
        return nullptr;
    }

    llama_model* model = static_cast<llama_model*>(model_handle);

    // Set up context parameters
    llama_context_params cparams = llama_context_default_params();

    // Create context
    llama_context* ctx = llama_new_context_with_model(model, cparams);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return nullptr;
    }

    return static_cast<llama_context_handle>(ctx);
}

const char* llama_binding_generate(llama_context_handle ctx_handle, const char* prompt,
                                int max_tokens, float temperature, float top_p, int top_k) {
    if (!ctx_handle) {
        return nullptr;
    }

    llama_context* ctx = static_cast<llama_context*>(ctx_handle);

    // Tokenize the prompt
    auto tokens = llama_tokenize(ctx, prompt, true);

    // Prepare context for generation
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
    }
    batch.logits[batch.n_tokens - 1] = 1;

    // Evaluate the prompt
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Failed to decode" << std::endl;
        return nullptr;
    }

    // Generate response
    std::stringstream ss;

    for (int i = 0; i < max_tokens; i++) {
        // Sample token
        llama_token token_id = llama_sample_token(ctx, NULL, temperature, top_p, top_k);

        // Check for special tokens (EOS)
        if (token_id == llama_token_eos(ctx)) {
            break;
        }

        // Convert token to text and append
        const char* token_text = llama_token_to_str(ctx, token_id);
        ss << token_text;

        // Add token to batch for next round
        batch = llama_batch_init(1, 0, 1);
        batch.token[0] = token_id;
        batch.pos[0] = tokens.size() + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Failed to decode" << std::endl;
            return nullptr;
        }
    }

    // Store the result in thread_local storage to ensure it remains valid
    generation_result = ss.str();
    return generation_result.c_str();
}

void llama_binding_free_context(llama_context_handle ctx_handle) {
    if (ctx_handle) {
        llama_context* ctx = static_cast<llama_context*>(ctx_handle);
        llama_free(ctx);
    }
}

void llama_binding_free_model(llama_model_handle model_handle) {
    if (model_handle) {
        llama_model* model = static_cast<llama_model*>(model_handle);
        llama_free_model(model);
    }
}
EOF

# Create Go binding
cat > "$INTEGRATION_DIR/llama.go" << 'EOF'
package llama_integration

// #cgo CXXFLAGS: -std=c++11
// #cgo LDFLAGS: -lstdc++
// #include <stdlib.h>
// #include "binding.h"
import "C"
import (
	"errors"
	"runtime"
	"unsafe"
)

func init() {
	// Initialize llama backend on package load
	C.llama_binding_initialize()
}

type LlamaModel struct {
	model   C.llama_model_handle
	context C.llama_context_handle
}

// NewLlamaModel creates a new LlamaModel instance with the given parameters
func NewLlamaModel(modelPath string, contextSize int, seed uint32, threads int) (*LlamaModel, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	// Load the model
	model := C.llama_binding_load_model(cModelPath, C.int(contextSize), C.uint32_t(seed), C.int(threads))
	if model == nil {
		return nil, errors.New("failed to load model")
	}

	// Create a context
	context := C.llama_binding_create_context(model)
	if context == nil {
		C.llama_binding_free_model(model)
		return nil, errors.New("failed to create context")
	}

	instance := &LlamaModel{
		model:   model,
		context: context,
	}

	// Set up finalizer to free resources
	runtime.SetFinalizer(instance, func(m *LlamaModel) {
		m.Free()
	})

	return instance, nil
}

// Generate generates text based on the provided prompt and parameters
func (m *LlamaModel) Generate(prompt string, maxTokens int, temperature float32, topP float32, topK int) (string, error) {
	if m.model == nil || m.context == nil {
		return "", errors.New("model not initialized")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	result := C.llama_binding_generate(
		m.context,
		cPrompt,
		C.int(maxTokens),
		C.float(temperature),
		C.float(topP),
		C.int(topK),
	)

	if result == nil {
		return "", errors.New("failed to generate text")
	}

	return C.GoString(result), nil
}

// Free releases resources used by the model
func (m *LlamaModel) Free() {
	if m.context != nil {
		C.llama_binding_free_context(m.context)
		m.context = nil
	}

	if m.model != nil {
		C.llama_binding_free_model(m.model)
		m.model = nil
	}
}
EOF

# Create build script for C++ binding
cat > "$INTEGRATION_DIR/build.sh" << 'EOF'
#!/bin/bash
set -e

ROOT_DIR=$(cd ../../../; pwd)
LLAMA_CPP_DIR="$ROOT_DIR/vendor/go-llama.cpp/llama.cpp"

# Compile the binding as a shared library
g++ -std=c++11 -fPIC -c binding.cpp -o binding.o \
    -I$LLAMA_CPP_DIR \
    -I$LLAMA_CPP_DIR/common

# On macOS, create a dynamic library
if [ "$(uname)" == "Darwin" ]; then
    g++ -shared -o libbinding.dylib binding.o \
        -L$LLAMA_CPP_DIR/build -lllama \
        -framework Accelerate
else
    # On Linux, create a .so file
    g++ -shared -o libbinding.so binding.o \
        -L$LLAMA_CPP_DIR/build -lllama
fi

echo "Built binding library successfully"
EOF

# Make the build script executable
chmod +x "$INTEGRATION_DIR/build.sh"

# Update Go integration wrapper to use our direct binding
cat > "$ROOT_DIR/pkg/llm/direct_llama_impl.go" << 'EOF'
//go:build cgo
// +build cgo

package llm

import (
	"errors"
	"fmt"
	"path/filepath"

	"github.com/andrew/llm-rag-poc/pkg/llm/llama_integration"
)

// DirectLlamaModel is an implementation of LLModel using our direct llama.cpp binding
type DirectLlamaModel struct {
	model *llama_integration.LlamaModel
}

// NewDirectLlamaModel creates a new DirectLlamaModel instance
func NewDirectLlamaModel(config LlamaConfig) (*DirectLlamaModel, error) {
	// Get absolute path to the model file
	absPath, err := filepath.Abs(config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to get absolute path to model: %w", err)
	}

	// Create llama model using our direct binding
	model, err := llama_integration.NewLlamaModel(
		absPath,
		config.ContextSize,
		uint32(config.Seed),
		config.Threads,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to initialize llama model: %w", err)
	}

	return &DirectLlamaModel{
		model: model,
	}, nil
}

// Predict implements the LLModel interface
func (m *DirectLlamaModel) Predict(text string, options ...PredictOption) (string, error) {
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

	topK := 40

	// Call the model's Generate method
	return m.model.Generate(
		text,
		tokens,
		temp,
		topP,
		topK,
	)
}

// Embeddings implements the LLModel interface
func (m *DirectLlamaModel) Embeddings(text string) ([]float32, error) {
	// Our simplified binding doesn't support embeddings yet
	return nil, errors.New("embeddings not currently supported by the direct binding")
}

// Close implements the LLModel interface
func (m *DirectLlamaModel) Close() error {
	if m.model == nil {
		return nil
	}
	m.model.Free()
	return nil
}

// CreateDirectLlamaClient creates a LlamaClient with our DirectLlamaModel implementation
func CreateDirectLlamaClient(config LlamaConfig) (*LlamaClient, error) {
	model, err := NewDirectLlamaModel(config)
	if err != nil {
		return nil, err
	}

	return &LlamaClient{
		model:  model,
		config: config,
	}, nil
}
EOF

# Update main.go to use our direct implementation
cat > "$ROOT_DIR/cmd/chat/main.go" << 'EOFmain'
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/andrew/llm-rag-poc/pkg/llm"
	"github.com/andrew/llm-rag-poc/pkg/models"
	"github.com/fatih/color"
)

var (
	modelPath    = flag.String("model", "models/llama3.2.gguf", "Path to the LLaMA model file")
	contextSize  = flag.Int("ctx", 2048, "Context size for the model")
	temperature  = flag.Float64("temp", 0.7, "Temperature for sampling")
	maxTokens    = flag.Int("max-tokens", 2048, "Maximum number of tokens to generate")
	threads      = flag.Int("threads", 4, "Number of threads to use")
	systemPrompt = flag.String("system", "You are a helpful, honest, and concise assistant.", "System prompt")
	gpuLayers    = flag.Int("ngl", 1, "Number of GPU layers to use (0 for CPU only)")
	metalShader  = flag.String("metal-shader", "./ggml-metal.metal", "Path to Metal shader (macOS only)")
)

func main() {
	flag.Parse()

	// Initialize context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle interrupts
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Println("\nShutting down...")
		cancel()
		os.Exit(0)
	}()

	// Get absolute path to model
	absModelPath, err := filepath.Abs(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error resolving model path: %v\n", err)
		os.Exit(1)
	}

	// Get absolute path to metal shader if provided
	absMetalShader, err := filepath.Abs(*metalShader)
	if err != nil {
		// Not fatal, will fall back to CPU if needed
		fmt.Printf("Warning: Could not resolve metal shader path: %v\n", err)
		absMetalShader = ""
	}

	// Configure and initialize the LLM client
	config := llm.DefaultLlamaConfig()
	config.ModelPath = absModelPath
	config.ContextSize = *contextSize
	config.Threads = *threads
	config.UseNGL = *gpuLayers

	// Only set metal shader path if it exists
	if _, err := os.Stat(absMetalShader); err == nil {
		config.UseMetalShaderPath = absMetalShader
	} else {
		fmt.Printf("Warning: Metal shader not found at %s, falling back to CPU\n", absMetalShader)
	}

	// Verify that the model file exists
	if _, err := os.Stat(absModelPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error: Model file not found at %s\n", absModelPath)
		os.Exit(1)
	}

	fmt.Printf("Initializing LLM with model: %s\n", absModelPath)

	// Use our direct llama implementation
	client, err := llm.CreateDirectLlamaClient(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing LLM: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// Print welcome message
	boldGreen := color.New(color.FgGreen, color.Bold).SprintFunc()
	boldCyan := color.New(color.FgCyan, color.Bold).SprintFunc()
	fmt.Println(boldGreen("🦙 LLM RAG Chat Interface"))
	fmt.Printf("Using model: %s\n", boldCyan(absModelPath))
	fmt.Printf("Threads: %d, Context Size: %d, GPU Layers: %d\n",
		*threads, *contextSize, *gpuLayers)
	fmt.Println("Type your message and press Enter. Type 'exit' or press Ctrl+C to quit.")
	fmt.Println()

	// Start chat session
	conversation := []models.Message{
		{
			Role:      models.RoleSystem,
			Content:   *systemPrompt,
			Timestamp: time.Now(),
		},
	}

	scanner := bufio.NewScanner(os.Stdin)
	modelConfig := llm.ModelConfig{
		Temperature: float32(*temperature),
		MaxTokens:   *maxTokens,
	}

	for {
		// Get user input
		fmt.Print(boldGreen("You: "))
		if !scanner.Scan() {
			break
		}
		userInput := scanner.Text()

		// Check for exit command
		if strings.ToLower(strings.TrimSpace(userInput)) == "exit" {
			break
		}

		// Add user message to conversation
		userMessage := models.Message{
			Role:      models.RoleUser,
			Content:   userInput,
			Timestamp: time.Now(),
		}
		conversation = append(conversation, userMessage)

		// Generate response
		fmt.Print(boldCyan("Assistant: "))
		response, err := client.Chat(ctx, conversation, modelConfig)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			continue
		}

		// Output response
		fmt.Println(response.Content)
		fmt.Println()

		// Add assistant response to conversation history
		conversation = append(conversation, response)
	}
}
EOFmain

echo "Direct llama.cpp integration setup complete!"
echo "Run the following commands to build and run:"
echo "  cd pkg/llm/llama_integration && ./build.sh"
echo "  cd ../../../"
echo "  CGO_LDFLAGS=\"-L\$PWD/pkg/llm/llama_integration -lbinding\" go build -tags cgo -o bin/chat ./cmd/chat"
echo "  ./bin/chat --model models/llama3.2.gguf"
