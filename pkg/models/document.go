package models

import "time"

// Document represents a document that can be indexed for RAG
type Document struct {
	ID          string            `json:"id"`
	Content     string            `json:"content"`
	Metadata    map[string]string `json:"metadata"`
	Source      string            `json:"source"`
	Created     time.Time         `json:"created"`
	LastUpdated time.Time         `json:"last_updated"`
}

// Chunk represents a smaller piece of a document that can be vectorized
type Chunk struct {
	ID         string            `json:"id"`
	DocumentID string            `json:"document_id"`
	Content    string            `json:"content"`
	Metadata   map[string]string `json:"metadata"`
	Embedding  []float32         `json:"embedding,omitempty"`
}

// SearchResult represents a document chunk that matched a query
type SearchResult struct {
	Chunk       Chunk     `json:"chunk"`
	Score       float32   `json:"score"`
	RetrievedAt time.Time `json:"retrieved_at"`
}
