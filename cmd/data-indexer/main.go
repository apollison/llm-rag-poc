package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/andrew/llm-rag-poc/pkg/llm"
	"github.com/andrew/llm-rag-poc/pkg/models"
	"github.com/google/uuid"
)

type IndexRequest struct {
	FilePath string            `json:"file_path"`
	Text     string            `json:"text,omitempty"`
	URL      string            `json:"url,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

type ChunkResponse struct {
	DocumentID string         `json:"document_id"`
	Chunks     []models.Chunk `json:"chunks"`
}

var (
	port         = flag.Int("port", 8081, "Port to listen on")
	modelPath    = flag.String("model", "models/llama-3-8b.gguf", "Path to the embedding model file")
	chunkSize    = flag.Int("chunk-size", 512, "Size of text chunks for indexing")
	chunkOverlap = flag.Int("chunk-overlap", 128, "Overlap between text chunks")
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

	// Configure and initialize the LLM client for embeddings
	config := llm.DefaultLlamaConfig()
	config.ModelPath = *modelPath

	client, err := llm.NewLlamaClient(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing LLM: %v\n", err)
		os.Exit(1)
	}
	defer client.Close()

	// Define HTTP handlers
	http.HandleFunc("/index/file", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req IndexRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if req.FilePath == "" {
			http.Error(w, "File path is required", http.StatusBadRequest)
			return
		}

		// Process the file
		documentID, chunks, err := processFile(req.FilePath, req.Metadata)
		if err != nil {
			http.Error(w, fmt.Sprintf("Error processing file: %v", err), http.StatusInternalServerError)
			return
		}

		// TODO: Generate embeddings for chunks and store them in vector database

		// Return the processed chunks
		resp := ChunkResponse{
			DocumentID: documentID,
			Chunks:     chunks,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "Error encoding response", http.StatusInternalServerError)
			return
		}
	})

	http.HandleFunc("/index/text", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req IndexRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if req.Text == "" {
			http.Error(w, "Text content is required", http.StatusBadRequest)
			return
		}

		// Process the text
		documentID := uuid.New().String()
		chunks := chunkText(req.Text, *chunkSize, *chunkOverlap)

		// Create document chunks
		var modelChunks []models.Chunk
		for i, chunk := range chunks {
			chunkID := fmt.Sprintf("%s-chunk-%d", documentID, i)
			modelChunks = append(modelChunks, models.Chunk{
				ID:         chunkID,
				DocumentID: documentID,
				Content:    chunk,
				Metadata:   req.Metadata,
			})
		}

		// TODO: Generate embeddings for chunks and store them in vector database

		// Return the processed chunks
		resp := ChunkResponse{
			DocumentID: documentID,
			Chunks:     modelChunks,
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
		log.Printf("Starting data indexer service on port %d\n", *port)
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

// processFile reads a file and splits it into chunks
func processFile(path string, metadata map[string]string) (string, []models.Chunk, error) {
	// Check if file exists
	info, err := os.Stat(path)
	if err != nil {
		return "", nil, fmt.Errorf("file not found: %v", err)
	}

	// Don't process directories
	if info.IsDir() {
		return "", nil, fmt.Errorf("path is a directory, not a file")
	}

	// Read the file content
	file, err := os.Open(path)
	if err != nil {
		return "", nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	// Read all content
	var content strings.Builder
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		content.WriteString(scanner.Text())
		content.WriteString("\n")
	}

	if err := scanner.Err(); err != nil {
		return "", nil, fmt.Errorf("error reading file: %v", err)
	}

	// Create a document ID
	documentID := uuid.New().String()

	// If metadata is nil, initialize it
	if metadata == nil {
		metadata = make(map[string]string)
	}

	// Add filename to metadata
	metadata["filename"] = filepath.Base(path)
	metadata["filepath"] = path

	// Chunk the content
	chunks := chunkText(content.String(), *chunkSize, *chunkOverlap)

	// Create model chunks
	var modelChunks []models.Chunk
	for i, chunk := range chunks {
		chunkID := fmt.Sprintf("%s-chunk-%d", documentID, i)
		modelChunks = append(modelChunks, models.Chunk{
			ID:         chunkID,
			DocumentID: documentID,
			Content:    chunk,
			Metadata:   metadata,
		})
	}

	return documentID, modelChunks, nil
}

// chunkText splits text into chunks with optional overlap
func chunkText(text string, chunkSize, overlap int) []string {
	if len(text) == 0 {
		return []string{}
	}

	// Simple text splitting by characters
	// In a real implementation, you'd want smarter chunking that respects sentence/paragraph boundaries
	var chunks []string

	// Handle case where text is shorter than chunk size
	if len(text) <= chunkSize {
		return []string{text}
	}

	// Split text into chunks
	for i := 0; i < len(text); i += (chunkSize - overlap) {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}

		// Add the chunk
		chunks = append(chunks, text[i:end])

		// If we've reached the end of the text
		if end == len(text) {
			break
		}
	}

	return chunks
}
