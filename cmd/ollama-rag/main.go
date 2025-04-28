package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

// logError logs an error with file and line information
func logError(err error) {
	_, file, line, _ := runtime.Caller(1)
	log.Fatalf("😡 %s:%d - %v", file, line, err)
}

// logDebug prints debug information only when debug mode is enabled
func logDebug(format string, args ...interface{}) {
	if debugMode {
		fmt.Printf(format+"\n", args...)
	}
}

var (
	FALSE     = false
	TRUE      = true
	debugMode = false // Global debug flag
)

// System instructions used across all modes
const SYSTEM_INSTRUCTIONS = `You are a helpful assistant.
You offer stock and option trading advice.
You are a financial advisor, and offer stock and options trading advice.
Respond with specific trading recommendations when asked.
You specialize in the wheel strategy and options trading.`

func main() {
	// Parse command-line flags
	interactive := flag.Bool("interactive", false, "Run in interactive chat mode")
	debug := flag.Bool("debug", false, "Enable debug output")
	flag.Parse()

	// Set the global debug mode
	debugMode = *debug

	ctx := context.Background()

	// Set up Ollama client
	client := setupOllamaClient()

	// Load and prepare knowledge base
	vectorStore := setupKnowledgeBase(ctx, client)

	if *interactive {
		runInteractiveMode(ctx, client, vectorStore)
	} else {
		runSingleQuestionMode(ctx, client, vectorStore)
	}
}

// setupOllamaClient configures and tests the Ollama API client
func setupOllamaClient() *api.Client {
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
	logDebug("🔄 Connecting to Ollama server at %s", ollamaRawUrl)
	resp, err := http.Get(ollamaRawUrl + "/api/tags")
	if err != nil {
		logDebug("⚠️ Warning: Ollama server might not be running at %s", ollamaRawUrl)
		logDebug("⚠️ Error: %v", err)
		logDebug("⚠️ Make sure the Ollama server is running before continuing.")
		// We'll exit with a meaningful error message
		logError(fmt.Errorf("cannot connect to Ollama server at %s: %w", ollamaRawUrl, err))
	} else {
		resp.Body.Close() // Clean up the response
		logDebug("✅ Successfully connected to Ollama server")
	}

	return client
}

// setupKnowledgeBase loads and embeds the content for RAG
func setupKnowledgeBase(ctx context.Context, client *api.Client) []VectorRecord {
	// Define the root content directory
	contentDir := "../../content"

	// Get all content files recursively
	contentFiles, err := getAllContentFiles(contentDir)
	if err != nil {
		logError(fmt.Errorf("error finding content files: %w", err))
	}

	logDebug("📚 Found %d content files to process", len(contentFiles))

	// Collect content from all files
	var allContent strings.Builder

	for _, filePath := range contentFiles {
		contentBytes, err := os.ReadFile(filePath)
		if err != nil {
			logDebug("⚠️ Error reading file %s: %v", filePath, err)
			continue
		}

		// Add metadata header to identify the source file
		relPath, _ := filepath.Rel(contentDir, filePath)
		header := fmt.Sprintf("\n# SOURCE: %s\n\n", relPath)
		allContent.WriteString(header)
		allContent.Write(contentBytes)
		allContent.WriteString("\n\n")

		logDebug("📄 Loaded content from: %s (%d bytes)", relPath, len(contentBytes))
	}

	content := allContent.String()
	logDebug("📄 Total content size: %d bytes", len(content))

	// Use smaller chunks with less overlap to prevent embedding issues
	// Maximum recommended size for embedding is around 512 tokens (roughly 2048 chars)
	chunks := ChunkText(content, 400, 50)
	logDebug("🧩 Split content into %d chunks", len(chunks))

	vectorStore := []VectorRecord{}
	// Create embeddings from documents and save them in the store
	for idx, chunk := range chunks {
		logDebug("📝 Creating embedding nb: %d (size: %d chars)", idx, len(chunk))

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

	return vectorStore
}

// getAllContentFiles recursively finds all content files in a directory
func getAllContentFiles(rootDir string) ([]string, error) {
	var files []string

	err := filepath.Walk(rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if info.IsDir() {
			return nil
		}

		// Filter for markdown files
		if filepath.Ext(path) == ".md" {
			files = append(files, path)
		}

		return nil
	})

	return files, err
}

// runSingleQuestionMode processes a single predefined question
func runSingleQuestionMode(ctx context.Context, client *api.Client, vectorStore []VectorRecord) {
	question := `Based on my portfolio and AAPL option chain data, which option should I sell to open?
	I want to follow the wheel strategy and tasty trade principles.
	`

	// Process the question with RAG
	processRagQuestion(ctx, client, vectorStore, SYSTEM_INSTRUCTIONS, question)
}

// runInteractiveMode runs an interactive chat session
func runInteractiveMode(ctx context.Context, client *api.Client, vectorStore []VectorRecord) {
	reader := bufio.NewReader(os.Stdin)
	chatHistory := []api.Message{
		{Role: "system", Content: SYSTEM_INSTRUCTIONS},
	}

	// Always display welcome messages in interactive mode
	fmt.Println("\n🤖 Welcome to the interactive RAG chat mode!")
	fmt.Println("📊 Ask me any questions about stocks and options trading.")
	fmt.Println("💡 Type 'exit' or 'quit' to end the session.")

	for {
		// Always display prompt for user input
		fmt.Print("\n👤 You: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			logError(err)
		}

		// Trim whitespace and newlines
		question := strings.TrimSpace(input)

		// Check for exit command
		if strings.ToLower(question) == "exit" || strings.ToLower(question) == "quit" {
			fmt.Println("\n👋 Goodbye!")
			break
		}

		if question == "" {
			continue
		}

		// Process the question and update chat history
		chatHistory = processRagQuestionWithHistory(ctx, client, vectorStore, chatHistory, question)
	}
}

// processRagQuestionWithHistory processes a question using RAG and maintains chat history
func processRagQuestionWithHistory(ctx context.Context, client *api.Client, vectorStore []VectorRecord,
	history []api.Message, question string) []api.Message {

	// Get embedding for the question
	embeddingFromQuestion, err := GetEmbeddingFromChunk(ctx, client, question)
	if err != nil {
		logError(err)
	}

	// Find similarities and retrieve relevant context
	newContext := findRelevantContext(vectorStore, embeddingFromQuestion)

	// Add the new question to history
	history = append(history, api.Message{Role: "user", Content: question})

	// Add context as a system message just before generating the response
	contextMessage := api.Message{Role: "system", Content: "CONTENT:\n" + newContext}

	// Create a copy of history with context inserted before the user's question
	augmentedHistory := make([]api.Message, 0, len(history)+1)

	// Add all system messages first
	for _, msg := range history {
		if msg.Role == "system" {
			augmentedHistory = append(augmentedHistory, msg)
		}
	}

	// Add RAG context
	augmentedHistory = append(augmentedHistory, contextMessage)

	// Add non-system messages
	for _, msg := range history {
		if msg.Role != "system" {
			augmentedHistory = append(augmentedHistory, msg)
		}
	}

	req := &api.ChatRequest{
		Model:    "llama3.2",
		Messages: augmentedHistory,
		Options: map[string]interface{}{
			"temperature":    0.0,
			"repeat_last_n":  2,
			"repeat_penalty": 1.8,
			"top_k":          10,
			"top_p":          0.5,
		},
		Stream: &TRUE,
	}

	// Always show the assistant prompt in interactive mode
	fmt.Print("\n🤖 Assistant: ")

	var responseContent strings.Builder
	err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
		// Always display the assistant's response
		fmt.Print(resp.Message.Content)
		responseContent.WriteString(resp.Message.Content)
		return nil
	})

	if err != nil {
		logError(err)
	}
	fmt.Println()

	// Add the assistant's response to the history
	history = append(history, api.Message{Role: "assistant", Content: responseContent.String()})

	return history
}

// processRagQuestion processes a single question with RAG (no history tracking)
func processRagQuestion(ctx context.Context, client *api.Client, vectorStore []VectorRecord,
	systemInstructions string, question string) {

	// Get embedding for the question
	embeddingFromQuestion, err := GetEmbeddingFromChunk(ctx, client, question)
	if err != nil {
		logError(err)
	}

	// Find similarities and retrieve relevant context
	newContext := findRelevantContext(vectorStore, embeddingFromQuestion)

	// Prompt construction
	messages := []api.Message{
		{Role: "system", Content: systemInstructions},
		{Role: "system", Content: "CONTENT:\n" + newContext},
		{Role: "user", Content: question},
	}

	req := &api.ChatRequest{
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

	// Question should be visible in all modes
	fmt.Println("🦄 Question:", question)
	fmt.Println("🤖 Answer:")

	err = client.Chat(ctx, req, func(resp api.ChatResponse) error {
		// Response should always be visible, regardless of debug mode
		fmt.Print(resp.Message.Content)
		return nil
	})

	if err != nil {
		logError(err)
	}
	fmt.Println("\n")
}

// findRelevantContext finds the most relevant context chunks for a question
func findRelevantContext(vectorStore []VectorRecord, questionEmbedding []float64) string {
	// Calculate cosine similarity between the question and each vector in the store
	similarities := []Similarity{}

	for _, vector := range vectorStore {
		cosineSimilarity, err := CosineSimilarity(questionEmbedding, vector.Embedding)
		if err != nil {
			logError(err)
		}

		// append to similarities
		similarities = append(similarities, Similarity{
			Prompt:           vector.Prompt,
			CosineSimilarity: cosineSimilarity,
		})
	}

	// Sort the similarities by cosine similarity
	sort.Slice(similarities, func(i, j int) bool {
		return similarities[i].CosineSimilarity > similarities[j].CosineSimilarity
	})

	// Take all similarities for now
	topSimilarities := similarities

	// Debug information
	logDebug("🔍 Top similarities:")
	for i, similarity := range topSimilarities {
		logDebug("🔍 [%d] Similarity: %.4f", i+1, similarity.CosineSimilarity)
		logDebug("--------------------------------------------------")
	}

	// Create a context with the top chunks
	newContext := ""
	for _, similarity := range topSimilarities {
		newContext += similarity.Prompt
	}

	return newContext
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

		logDebug("🔄 Embedding attempt %d/%d...", attempts+1, maxRetries)
		resp, err = client.Embeddings(reqCtx, req)

		if err == nil {
			logDebug("✅ Embedding request successful")
			return resp.Embedding, nil
		}

		lastErr = err
		retryDelay := time.Duration(math.Pow(2, float64(attempts))) * baseDelay
		logDebug("⚠️ Attempt %d/%d failed: %v (retrying in %v)",
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
		logDebug("⚠️ Document truncated to %d characters", maxSize)
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
