package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
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
	debugMode = false // Global debug flag
)

const (
	// Collection name for our RAG data in Qdrant
	COLLECTION_NAME = "stonks_rag"
	// Vector dimension size based on the llama3 model
	VECTOR_SIZE = 4096
)

func main() {
	// Parse command-line flags
	debug := flag.Bool("debug", false, "Enable debug output")
	qdrantHost := flag.String("qdrant-host", "localhost", "Qdrant server host")
	qdrantPort := flag.Int("qdrant-port", 6334, "Qdrant server gRPC port")
	contentDir := flag.String("content-dir", "../../content", "Directory containing content files")
	recreateCollection := flag.Bool("recreate", false, "Recreate the collection if it exists")
	flag.Parse()

	// Set the global debug mode
	debugMode = *debug

	// Setup context
	ctx := context.Background()

	// Connect to Qdrant
	connectStr := fmt.Sprintf("%s:%d", *qdrantHost, *qdrantPort)
	conn, err := grpc.Dial(connectStr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logError(fmt.Errorf("failed to connect to Qdrant: %w", err))
	}
	defer conn.Close()

	collectionsClient := qdrantclient.NewCollectionsClient(conn)
	pointsClient := qdrantclient.NewPointsClient(conn)

	logDebug("✅ Connected to Qdrant at %s:%d", *qdrantHost, *qdrantPort)

	// Set up Ollama client for creating embeddings
	ollamaClient := setupOllamaClient()

	// Create or recreate collection
	err = setupCollection(ctx, collectionsClient, *recreateCollection)
	if err != nil {
		logError(fmt.Errorf("failed to setup collection: %w", err))
	}

	// Index content files
	err = indexContentFiles(ctx, ollamaClient, pointsClient, *contentDir)
	if err != nil {
		logError(fmt.Errorf("indexing failed: %w", err))
	}

	fmt.Println("✅ Content successfully indexed in Qdrant!")
}

// setupCollection creates or recreates the vector collection in Qdrant
func setupCollection(ctx context.Context, client qdrantclient.CollectionsClient, recreate bool) error {
	// Check if collection exists
	collections, err := client.List(ctx, &qdrantclient.ListCollectionsRequest{})
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

	// If collection exists and recreate is true, delete it
	if exists && recreate {
		logDebug("🗑️ Deleting existing collection: %s", COLLECTION_NAME)
		_, err := client.Delete(ctx, &qdrantclient.DeleteCollection{
			CollectionName: COLLECTION_NAME,
		})
		if err != nil {
			return fmt.Errorf("failed to delete collection: %w", err)
		}
		exists = false
	}

	// Create collection if it doesn't exist
	if !exists {
		logDebug("🆕 Creating new collection: %s", COLLECTION_NAME)
		createReq := &qdrantclient.CreateCollection{
			CollectionName: COLLECTION_NAME,
			VectorsConfig: &qdrantclient.VectorsConfig{
				Config: &qdrantclient.VectorsConfig_Params{
					Params: &qdrantclient.VectorParams{
						Size:     uint64(VECTOR_SIZE),
						Distance: qdrantclient.Distance_Cosine,
					},
				},
			},
		}
		_, err = client.Create(ctx, createReq)
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}

		logDebug("✅ Collection '%s' created successfully", COLLECTION_NAME)
	}

	return nil
}

// indexContentFiles processes all content files and adds them to Qdrant
func indexContentFiles(ctx context.Context, ollamaClient *api.Client, pointsClient qdrantclient.PointsClient, contentDir string) error {
	// Get all content files recursively
	contentFiles, err := getAllContentFiles(contentDir)
	if err != nil {
		return fmt.Errorf("error finding content files: %w", err)
	}

	fmt.Printf("📚 Processing %d content files\n", len(contentFiles))

	var totalChunks int
	var totalPoints int

	// Process each file
	for fileIndex, filePath := range contentFiles {
		contentBytes, err := os.ReadFile(filePath)
		if err != nil {
			logDebug("⚠️ Error reading file %s: %v", filePath, err)
			continue
		}

		relPath, _ := filepath.Rel(contentDir, filePath)
		fmt.Printf("📄 [%d/%d] Processing: %s (%d bytes)\n", fileIndex+1, len(contentFiles), relPath, len(contentBytes))

		// Add metadata header to identify the source file
		content := string(contentBytes)

		// Use smaller chunks with less overlap to prevent embedding issues
		chunks := ChunkText(content, 400, 50)
		logDebug("🧩 Split into %d chunks", len(chunks))
		totalChunks += len(chunks)

		pointsToUpsert := make([]*qdrantclient.PointStruct, 0, len(chunks))

		// Process each chunk
		for chunkIndex, chunk := range chunks {
			logDebug("📝 Processing chunk %d/%d (size: %d chars)", chunkIndex+1, len(chunks), len(chunk))

			// Create embedding for the chunk
			embedding, err := SafeGetEmbeddingFromChunk(ctx, ollamaClient, chunk)
			if err != nil {
				logDebug("⚠️ Failed to create embedding: %v", err)
				continue
			}

			// Convert embedding to vector format for Qdrant
			vector := make([]float32, len(embedding))
			for i, val := range embedding {
				vector[i] = float32(val)
			}

			// Create a unique ID for this point
			pointId := uint64(totalPoints + 1)
			totalPoints++

			// Create the point with payload
			point := &qdrantclient.PointStruct{
				Id: &qdrantclient.PointId{
					PointIdOptions: &qdrantclient.PointId_Num{
						Num: pointId,
					},
				},
				Vectors: &qdrantclient.Vectors{
					VectorsOptions: &qdrantclient.Vectors_Vector{
						Vector: &qdrantclient.Vector{
							Data: vector,
						},
					},
				},
				Payload: map[string]*qdrantclient.Value{
					"text":   {Kind: &qdrantclient.Value_StringValue{StringValue: chunk}},
					"source": {Kind: &qdrantclient.Value_StringValue{StringValue: relPath}},
				},
			}

			pointsToUpsert = append(pointsToUpsert, point)

			// Batch upsert every 100 points or on the last chunk
			if len(pointsToUpsert) >= 100 || (fileIndex == len(contentFiles)-1 && chunkIndex == len(chunks)-1) {
				if len(pointsToUpsert) > 0 {
					logDebug("📤 Upserting batch of %d points", len(pointsToUpsert))
					upsertReq := &qdrantclient.UpsertPoints{
						CollectionName: COLLECTION_NAME,
						Points:         pointsToUpsert,
					}
					_, err = pointsClient.Upsert(ctx, upsertReq)
					if err != nil {
						return fmt.Errorf("failed to upsert points: %w", err)
					}
					pointsToUpsert = pointsToUpsert[:0]
				}
			}
		}
	}

	fmt.Printf("✅ Indexed %d chunks from %d files into Qdrant\n", totalChunks, len(contentFiles))
	return nil
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

// ChunkText splits a text into chunks of specified size with overlap
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

// GetEmbeddingFromChunk gets an embedding vector for a text chunk
func GetEmbeddingFromChunk(ctx context.Context, client *api.Client, doc string) ([]float64, error) {
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
