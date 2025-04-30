package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	qdrantclient "github.com/qdrant/go-client/qdrant"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
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

const (
	// Collection name for our RAG data in Qdrant
	COLLECTION_NAME = "stonks_rag"
	// Limit for similarity search results
	SEARCH_LIMIT = 10
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
	qdrantHost := flag.String("qdrant-host", "localhost", "Qdrant server host")
	qdrantPort := flag.Int("qdrant-port", 6334, "Qdrant server gRPC port")
	flag.Parse()

	// Set the global debug mode
	debugMode = *debug

	ctx := context.Background()

	// Connect to Qdrant
	qdrantClient, err := connectToQdrant(*qdrantHost, *qdrantPort)
	if err != nil {
		logError(fmt.Errorf("failed to connect to Qdrant: %w", err))
	}
	logDebug("✅ Connected to Qdrant at %s:%d", *qdrantHost, *qdrantPort)

	// Verify Qdrant collection exists
	if err := verifyQdrantCollection(ctx, qdrantClient); err != nil {
		logError(fmt.Errorf("Qdrant collection issue: %w. Did you run the indexer first?", err))
	}

	// Set up Ollama client for embedding queries
	ollamaClient := setupOllamaClient()

	if *interactive {
		runInteractiveMode(ctx, ollamaClient, qdrantClient)
	} else {
		runSingleQuestionMode(ctx, ollamaClient, qdrantClient)
	}
}

// connectToQdrant establishes a connection to the Qdrant server
func connectToQdrant(host string, port int) (qdrantclient.CollectionsClient, error) {
	addr := fmt.Sprintf("%s:%d", host, port)
	conn, err := grpc.Dial(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Qdrant at %s: %w", addr, err)
	}

	return qdrantclient.NewCollectionsClient(conn), nil
}

// verifyQdrantCollection checks if the required collection exists in Qdrant
func verifyQdrantCollection(ctx context.Context, client qdrantclient.CollectionsClient) error {
	req := &qdrantclient.ListCollectionsRequest{}
	collections, err := client.List(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to list collections: %w", err)
	}

	exists := false
	for _, col := range collections.GetCollections() {
		if col.GetName() == COLLECTION_NAME {
			exists = true
			break
		}
	}

	if !exists {
		return fmt.Errorf("collection '%s' does not exist", COLLECTION_NAME)
	}

	logDebug("✅ Verified Qdrant collection '%s' exists", COLLECTION_NAME)
	return nil
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

// runSingleQuestionMode processes a single predefined question
func runSingleQuestionMode(ctx context.Context, ollamaClient *api.Client, qdrantClient qdrantclient.CollectionsClient) {
	question := `Based on my portfolio and AAPL option chain data, which option should I sell to open?
	I want to follow the wheel strategy and tasty trade principles.
	`

	// Process the question with RAG
	processRagQuestion(ctx, ollamaClient, qdrantClient, SYSTEM_INSTRUCTIONS, question, nil, false)
}

// runInteractiveMode runs an interactive chat session
func runInteractiveMode(ctx context.Context, ollamaClient *api.Client, qdrantClient qdrantclient.CollectionsClient) {
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
		response := processRagQuestion(ctx, ollamaClient, qdrantClient, SYSTEM_INSTRUCTIONS, question, chatHistory, true)
		chatHistory = append(chatHistory, api.Message{Role: "user", Content: question})
		chatHistory = append(chatHistory, api.Message{Role: "assistant", Content: response})
	}
}

// processRagQuestion processes a question with RAG and returns the response
// It can be used in both single question mode and interactive mode
func processRagQuestion(ctx context.Context, ollamaClient *api.Client, qdrantClient qdrantclient.CollectionsClient,
	systemInstructions string, question string, messages []api.Message, interactive bool) string {

	// Get embedding for the question
	embeddingFromQuestion, err := GetEmbeddingFromChunk(ctx, ollamaClient, question)
	if err != nil {
		logError(err)
	}

	// Find similar documents from Qdrant
	newContext, err := findRelevantContextFromQdrant(ctx, qdrantClient, embeddingFromQuestion)
	if err != nil {
		logError(err)
	}

	// If messages are provided, use them as the base (for interactive mode)
	// Otherwise, create a fresh message array (for single question mode)
	var finalMessages []api.Message
	if messages != nil {
		// Create a new slice for the messages to send to the LLM
		finalMessages = []api.Message{
			// Always include the system instructions first for consistent behavior
			{Role: "system", Content: systemInstructions},
			// Add RAG context second
			{Role: "system", Content: "CONTENT:\n" + newContext},
		}

		// Add conversation history (only user and assistant messages)
		for _, msg := range messages {
			if msg.Role != "system" {
				finalMessages = append(finalMessages, msg)
			}
		}
	} else {
		// For single question mode, use this simpler structure
		finalMessages = []api.Message{
			{Role: "system", Content: systemInstructions},
			{Role: "system", Content: "CONTENT:\n" + newContext},
			{Role: "user", Content: question},
		}
	}

	req := &api.ChatRequest{
		Model:    "llama3.2",
		Messages: finalMessages,
		Options: map[string]interface{}{
			"temperature":    0.0,
			"repeat_last_n":  2,
			"repeat_penalty": 1.8,
			"top_k":          10,
			"top_p":          0.5,
		},
		Stream: &TRUE,
	}

	// Output formatting depends on the mode
	if !interactive {
		fmt.Println("🦄 Question:", question)
		fmt.Println("🤖 Answer:")
	} else {
		fmt.Print("\n🤖 Assistant: ")
	}

	var responseContent strings.Builder
	err = ollamaClient.Chat(ctx, req, func(resp api.ChatResponse) error {
		fmt.Print(resp.Message.Content)
		responseContent.WriteString(resp.Message.Content)
		return nil
	})

	if err != nil {
		logError(err)
	}
	fmt.Println()

	return responseContent.String()
}

// findRelevantContextFromQdrant queries Qdrant for relevant context based on embeddings
func findRelevantContextFromQdrant(ctx context.Context, client qdrantclient.CollectionsClient, embedding []float64) (string, error) {
	// Get a points client for vector search
	// Since we can't directly access the connection, create a new connection with same params
	conn, ok := client.(interface {
		GetConnection() grpc.ClientConnInterface
	})
	if !ok {
		// If we can't get the connection from the client, create new one with host/port
		// This is a fallback that assumes standard connection params (localhost:6334)
		newConn, err := grpc.Dial("localhost:6334", grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			return "", fmt.Errorf("failed to create new connection to Qdrant: %w", err)
		}
		pointsClient := qdrantclient.NewPointsClient(newConn)
		return searchInQdrant(ctx, pointsClient, embedding)
	}

	pointsClient := qdrantclient.NewPointsClient(conn.GetConnection())
	return searchInQdrant(ctx, pointsClient, embedding)
}

// searchInQdrant performs the actual similarity search in Qdrant
func searchInQdrant(ctx context.Context, client qdrantclient.PointsClient, embedding []float64) (string, error) {
	// Convert the embedding to Qdrant vector format
	vector := make([]float32, len(embedding))
	for i, val := range embedding {
		vector[i] = float32(val)
	}

	// Search for similar vectors in Qdrant
	searchReq := &qdrantclient.SearchPoints{
		CollectionName: COLLECTION_NAME,
		Vector:         vector,
		Limit:          uint64(SEARCH_LIMIT),
		WithPayload: &qdrantclient.WithPayloadSelector{
			SelectorOptions: &qdrantclient.WithPayloadSelector_Include{
				Include: &qdrantclient.PayloadIncludeSelector{
					Fields: []string{"text", "source"},
				},
			},
		},
	}

	searchResp, err := client.Search(ctx, searchReq)
	if err != nil {
		return "", fmt.Errorf("failed to search in Qdrant: %w", err)
	}

	if len(searchResp.Result) == 0 {
		return "No relevant information found.", nil
	}

	// Debug information
	logDebug("🔍 Found %d relevant chunks in Qdrant", len(searchResp.Result))

	// Combine relevant chunks into a single context
	var contextBuilder strings.Builder

	for i, point := range searchResp.Result {
		score := point.GetScore()
		source := ""
		text := ""

		if sourceVal, ok := point.Payload["source"]; ok {
			source = sourceVal.GetStringValue()
		}

		if textVal, ok := point.Payload["text"]; ok {
			text = textVal.GetStringValue()
		}

		logDebug("🔍 [%d] Source: %s, Score: %.4f", i+1, source, score)

		// Add source information to the context
		if source != "" {
			contextBuilder.WriteString(fmt.Sprintf("# SOURCE: %s\n\n", source))
		}

		// Add the text content
		contextBuilder.WriteString(text)
		contextBuilder.WriteString("\n\n")
	}

	return contextBuilder.String(), nil
}

// GetEmbeddingFromChunk gets an embedding vector for a text chunk
func GetEmbeddingFromChunk(ctx context.Context, client *api.Client, doc string) ([]float64, error) {
	// Use llama3 as it's more commonly available on Ollama installations
	// This model should be consistent with what was used in the indexer
	embeddingsModel := "llama3"

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
