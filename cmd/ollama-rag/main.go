package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"sort"
	"time"

	"github.com/ollama/ollama/api"
)

// logError logs an error with file and line information
func logError(err error) {
	_, file, line, _ := runtime.Caller(1)
	log.Fatalf("😡 %s:%d - %v", file, line, err)
}

var (
	FALSE = false
	TRUE  = true
)

func main() {
	ctx := context.Background()

	var ollamaRawUrl string
	if ollamaRawUrl = os.Getenv("OLLAMA_HOST"); ollamaRawUrl == "" {
		ollamaRawUrl = "http://localhost:11434"
	}

	// Parse the URL properly
	ollamaUrl, err := url.Parse(ollamaRawUrl)
	if err != nil {
		logError(fmt.Errorf("invalid OLLAMA_HOST URL: %w", err))
	}

	// Create a custom HTTP client with timeout
	httpClient := &http.Client{
		Timeout: 30 * time.Second,
	}

	client := api.NewClient(ollamaUrl, httpClient)

	// Verify Ollama server is running
	fmt.Printf("🔄 Connecting to Ollama server at %s\n", ollamaRawUrl)
	resp, err := http.Get(ollamaRawUrl + "/api/tags")
	if err != nil {
		fmt.Printf("⚠️ Warning: Ollama server might not be running at %s\n", ollamaRawUrl)
		fmt.Printf("⚠️ Error: %v\n", err)
		fmt.Println("⚠️ Make sure the Ollama server is running before continuing.")
		// We'll exit with a meaningful error message
		logError(fmt.Errorf("cannot connect to Ollama server at %s: %w", ollamaRawUrl, err))
	} else {
		resp.Body.Close() // Clean up the response
		fmt.Println("✅ Successfully connected to Ollama server")
	}

	systemInstructions := `You are a helpful assistant.
	  You offer stock and option trading advice.
		Respond with a specific option to trade
		`

	question := `based on the AAPL option chain, which option should I sell to open?
	I want to start the wheel strategy.
	I do not have any positions on AAPL.
	`

	contentBytes, err := os.ReadFile("../../content/stonks.md")
	if err != nil {
		logError(err)
	}
	context := string(contentBytes)
	fmt.Printf("📄 Content size: %d bytes\n", len(context))

	// Use smaller chunks with less overlap to prevent embedding issues
	// Maximum recommended size for embedding is around 512 tokens (roughly 2048 chars)
	chunks := ChunkText(context, 400, 50)
	fmt.Printf("🧩 Split content into %d chunks\n", len(chunks))

	vectorStore := []VectorRecord{}
	// Create embeddings from documents and save them in the store
	for idx, chunk := range chunks {
		fmt.Printf("📝 Creating embedding nb: %d (size: %d chars)\n", idx, len(chunk))

		// Use SafeGetEmbeddingFromChunk which handles large text better
		embedding, err := SafeGetEmbeddingFromChunk(ctx, client, chunk)
		if err != nil {
			logError(err)
		}

		// Save the embedding in the vector store
		record := VectorRecord{
			Prompt:    chunk,
			Embedding: embedding,
		}
		vectorStore = append(vectorStore, record)
	}

	embeddingFromQuestion, err := GetEmbeddingFromChunk(ctx, client, question)
	if err != nil {
		logError(err)
	}

	// Search similarites between the question and the vectors of the store
	// 1- calculate the cosine similarity between the question and each vector in the store
	similarities := []Similarity{}

	for _, vector := range vectorStore {
		cosineSimilarity, err := CosineSimilarity(embeddingFromQuestion, vector.Embedding)
		if err != nil {
			logError(err)
		}

		// append to similarities
		similarities = append(similarities, Similarity{
			Prompt:           vector.Prompt,
			CosineSimilarity: cosineSimilarity,
		})
	}

	// Select the 5 most similar chunks
	// retrieve in similarities the 5 records with the highest cosine similarity
	// sort the similarities
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].CosineSimilarity > similarities[j].CosineSimilarity
	})

	// get the first 5 records
	// top5Similarities := similarities[:5]
	top5Similarities := similarities

	fmt.Println("🔍 Top 5 similarities:")
	for _, similarity := range top5Similarities {
		fmt.Println("🔍 Prompt:", similarity.Prompt)
		fmt.Println("🔍 Cosine similarity:", similarity.CosineSimilarity)
		fmt.Println("--------------------------------------------------")

	}

	// Create a new context with the top 5 chunks
	newContext := ""
	for _, similarity := range top5Similarities {
		newContext += similarity.Prompt
	}

	// Answer the question with the new context

	// Prompt construction
	messages := []api.Message{
		{Role: "system", Content: systemInstructions},
		{Role: "system", Content: "CONTENT:\n" + newContext},
		{Role: "user", Content: question},
	}

	req := &api.ChatRequest{
		// Model:    "qwen2.5:0.5b",
		Model:    "llama3.2",
		Messages: messages,
		Options: map[string]interface{}{
			"temperature":    0.0,
			"repeat_last_n":  2,
			"repeat_penalty": 1.8,
			"top_k":          10,
			"top_p":          0.5,
		},
		Stream: &TRUE,
	}

	fmt.Println("🦄 question:", question)
	fmt.Println("🤖 answer:")
	err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
		fmt.Print(resp.Message.Content)
		return nil
	})

	if err != nil {
		logError(err)
	}
	fmt.Println()
	fmt.Println()
}

func ChunkText(text string, chunkSize, overlap int) []string {
	chunks := []string{}
	for start := 0; start < len(text); start += chunkSize - overlap {
		end := start + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[start:end])
	}
	return chunks
}

type VectorRecord struct {
	Prompt    string    `json:"prompt"`
	Embedding []float64 `json:"embedding"`
}

type Similarity struct {
	Prompt           string
	CosineSimilarity float64
}

func GetEmbeddingFromChunk(ctx context.Context, client *api.Client, doc string) ([]float64, error) {
	embeddingsModel := "snowflake-arctic-embed:33m"

	req := &api.EmbeddingRequest{
		Model:  embeddingsModel,
		Prompt: doc,
	}

	var resp *api.EmbeddingResponse
	var err error
	var lastErr error

	// More robust retry logic with exponential backoff
	maxRetries := 3
	baseDelay := 1 * time.Second

	for attempts := 0; attempts < maxRetries; attempts++ {
		// Context with timeout for this specific request
		reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		fmt.Printf("🔄 Embedding attempt %d/%d...\n", attempts+1, maxRetries)
		resp, err = client.Embeddings(reqCtx, req)

		if err == nil {
			fmt.Println("✅ Embedding request successful")
			return resp.Embedding, nil
		}

		lastErr = err
		retryDelay := time.Duration(math.Pow(2, float64(attempts))) * baseDelay
		fmt.Printf("⚠️ Attempt %d/%d failed: %v (retrying in %v)\n",
			attempts+1, maxRetries, err, retryDelay)
		time.Sleep(retryDelay)
	}

	// If we get here, all attempts failed
	return nil, fmt.Errorf("embedding failed after %d attempts: %w", maxRetries, lastErr)
}

// SafeGetEmbeddingFromChunk handles large text better by truncating it
func SafeGetEmbeddingFromChunk(ctx context.Context, client *api.Client, doc string) ([]float64, error) {
	// Truncate the document if it exceeds the maximum recommended size
	maxSize := 2048
	if len(doc) > maxSize {
		doc = doc[:maxSize]
		fmt.Printf("⚠️ Document truncated to %d characters\n", maxSize)
	}
	return GetEmbeddingFromChunk(ctx, client, doc)
}

// CosineSimilarity calculates the cosine similarity between two vectors
// Returns a value between -1 and 1, where:
// 1 means vectors are identical
// 0 means vectors are perpendicular
// -1 means vectors are opposite
func CosineSimilarity(vec1, vec2 []float64) (float64, error) {
	if len(vec1) != len(vec2) {
		return 0, errors.New("vectors must have the same length")
	}

	// Calculate dot product
	dotProduct := 0.0
	magnitude1 := 0.0
	magnitude2 := 0.0

	for i := 0; i < len(vec1); i++ {
		dotProduct += vec1[i] * vec2[i]
		magnitude1 += vec1[i] * vec1[i]
		magnitude2 += vec2[i] * vec2[i]
	}

	magnitude1 = math.Sqrt(magnitude1)
	magnitude2 = math.Sqrt(magnitude2)

	// Check for zero magnitudes to avoid division by zero
	if magnitude1 == 0 || magnitude2 == 0 {
		return 0, errors.New("vector magnitude cannot be zero")
	}

	return dotProduct / (magnitude1 * magnitude2), nil
}
