package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/andrew/llm-rag-poc/pkg/llm"
	"github.com/andrew/llm-rag-poc/pkg/models"
)

type RagRequest struct {
	Query       string           `json:"query"`
	Messages    []models.Message `json:"messages,omitempty"`
	ModelConfig llm.ModelConfig  `json:"model_config,omitempty"`
}

type RagResponse struct {
	Answer       string                `json:"answer"`
	Citations    []models.SearchResult `json:"citations,omitempty"`
	ProcessingMs int64                 `json:"processing_ms"`
}

var (
	port          = flag.Int("port", 8080, "Port to listen on")
	modelPath     = flag.String("model", "models/llama-3-8b.gguf", "Path to the LLaMA model file")
	contextSize   = flag.Int("ctx", 2048, "Context size for the model")
	threads       = flag.Int("threads", 4, "Number of threads to use")
	maxRetrievals = flag.Int("max-retrievals", 5, "Maximum number of documents to retrieve")
	ragPrompt     = flag.String("rag-prompt", "Answer the question based on the following context:\n\nContext:\n%s\n\nQuestion: %s", "RAG prompt template")
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

	// Configure and initialize the LLM client
	config := llm.DefaultLlamaConfig()
	config.ModelPath = *modelPath
	config.ContextSize = *contextSize
	config.Threads = *threads

	client, err := llm.NewLlamaClient(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing LLM: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// TODO: Initialize the retrieval service
	// For now, we'll mock it but this would connect to your vector store

	http.HandleFunc("/rag", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req RagRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		start := time.Now()

		// TODO: Implement actual retrieval from vector store
		// This is where you'd connect to your retrieval service
		mockResults := []models.SearchResult{
			{
				Chunk: models.Chunk{
					Content:  "This is a mock result for demonstration purposes.",
					Metadata: map[string]string{"source": "mock_document"},
				},
				Score: 0.95,
			},
		}

		// Format RAG prompt with retrieved context
		context := "This is a mock context for demonstration purposes."
		raggedPrompt := fmt.Sprintf(*ragPrompt, context, req.Query)

		// Generate response using the LLM
		modelConfig := req.ModelConfig
		if modelConfig.MaxTokens == 0 {
			modelConfig = llm.DefaultConfig()
		}

		answer, err := client.Generate(ctx, raggedPrompt, modelConfig)
		if err != nil {
			http.Error(w, fmt.Sprintf("Error generating response: %v", err), http.StatusInternalServerError)
			return
		}

		// Prepare and send the response
		resp := RagResponse{
			Answer:       answer,
			Citations:    mockResults,
			ProcessingMs: time.Since(start).Milliseconds(),
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "Error encoding response", http.StatusInternalServerError)
			return
		}
	})

	// Start the HTTP server
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", *port),
		Handler: nil,
	}

	go func() {
		log.Printf("Starting RAG service on port %d\n", *port)
		if err := server.ListenAndServe(); err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	// Wait for interrupt signal
	<-ctx.Done()

	// Shutdown the server gracefully
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("Server shutdown failed: %v", err)
	}
}
