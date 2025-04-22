package vector

import (
	"context"

	"github.com/andrew/llm-rag-poc/pkg/models"
)

// Store defines the interface for vector database operations
type Store interface {
	// AddVector inserts or updates a vector in the database
	AddVector(ctx context.Context, id string, vector []float32, metadata map[string]string) error

	// Search finds the most similar vectors to the given query vector
	Search(ctx context.Context, queryVector []float32, limit int) ([]models.SearchResult, error)

	// Delete removes a vector from the store
	Delete(ctx context.Context, id string) error

	// Close releases resources used by the vector store
	Close() error
}

// Config contains configuration for a vector database
type Config struct {
	Type          string            // Type of vector database (e.g., "memory", "pgvector", "qdrant")
	Dimension     int               // Vector dimension size
	ConnectionURL string            // URL for database connection if applicable
	Options       map[string]string // Additional options
}
