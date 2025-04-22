package retrieval

import (
	"context"

	"github.com/andrew/llm-rag-poc/pkg/models"
)

// Service provides functionality for retrieving relevant documents
type Service interface {
	// SearchByText retrieves documents similar to the provided text query
	SearchByText(ctx context.Context, query string, limit int) ([]models.SearchResult, error)

	// SearchByVector retrieves documents similar to the provided vector
	SearchByVector(ctx context.Context, vector []float32, limit int) ([]models.SearchResult, error)

	// GetRetrievalContext generates a context string from search results for augmenting LLM prompts
	GetRetrievalContext(results []models.SearchResult) string
}

// Config contains configuration for a retrieval service
type Config struct {
	// MaxResults is the maximum number of results to return
	MaxResults int

	// ScoreThreshold is the minimum similarity score for results
	ScoreThreshold float32

	// ContextPrompt is the template for formatting retrieved context
	ContextPrompt string
}
