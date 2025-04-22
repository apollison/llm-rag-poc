# LLM RAG PoC

A playground for LLM (Large Language Model) applications with RAG (Retrieval Augmented Generation) capabilities, focusing on financial data and stock market information.

## Project Overview

This project implements a modular Go-based system for working with LLMs and RAG. The system consists of three main components:

1. **Chat Interface**: An interactive command-line chat application similar to Ollama
2. **RAG Service**: A service that enhances LLM responses with relevant context from a document store
3. **Data Indexer**: A service for processing, chunking, and indexing documents for retrieval

## Components

### Chat Interface

A command-line interface for direct interaction with LLMs.

```bash
go run cmd/chat/main.go --model models/llama-3-8b.gguf
```

Options:
- `--model`: Path to the LLaMA model file (default: models/llama-3-8b.gguf)
- `--ctx`: Context size for the model (default: 2048)
- `--temp`: Temperature for sampling (default: 0.7)
- `--max-tokens`: Maximum number of tokens to generate (default: 2048)
- `--threads`: Number of threads to use (default: 4)
- `--system`: System prompt (default: "You are a helpful, honest, and concise assistant.")

### RAG Service

A service that combines LLM capabilities with document retrieval for enhanced responses.

```bash
go run cmd/rag-service/main.go --model models/llama-3-8b.gguf
```

Options:
- `--port`: Port to listen on (default: 8080)
- `--model`: Path to the LLaMA model file (default: models/llama-3-8b.gguf)
- `--ctx`: Context size for the model (default: 2048)
- `--threads`: Number of threads to use (default: 4)
- `--max-retrievals`: Maximum number of documents to retrieve (default: 5)

### Data Indexer

A service for processing and indexing documents to be used in RAG.

```bash
go run cmd/data-indexer/main.go --model models/llama-3-8b.gguf
```

Options:
- `--port`: Port to listen on (default: 8081)
- `--model`: Path to the embedding model file (default: models/llama-3-8b.gguf)
- `--chunk-size`: Size of text chunks for indexing (default: 512)
- `--chunk-overlap`: Overlap between text chunks (default: 128)

## Getting Started

1. Install dependencies:
   ```bash
   go mod tidy
   ```

2. Download a language model (e.g., Llama 3):
   ```bash
   mkdir -p models
   # Download your preferred GGUF model and place it in the models directory
   ```

3. Run the chat interface:
   ```bash
   go run cmd/chat/main.go
   ```

## API Endpoints

### RAG Service (port 8080)

- `POST /rag`: Submit a query to retrieve context-enhanced responses

Request:
```json
{
  "query": "What were the trends in tech stocks last quarter?",
  "messages": [],
  "model_config": {
    "temperature": 0.7,
    "max_tokens": 1024
  }
}
```

### Data Indexer (port 8081)

- `POST /index/file`: Index a file
  ```json
  {
    "file_path": "/path/to/document.pdf",
    "metadata": {
      "source": "financial_report",
      "date": "2025-01-15"
    }
  }
  ```

- `POST /index/text`: Index raw text
  ```json
  {
    "text": "Tesla announced record profits...",
    "metadata": {
      "source": "news_article",
      "date": "2025-03-01"
    }
  }
  ```

## Project Structure

```
└── llm-rag-poc/
    ├── cmd/
    │   ├── chat/          # Chat interface binary
    │   ├── data-indexer/  # Document processing and indexing binary
    │   └── rag-service/   # RAG service binary
    └── pkg/
        ├── llm/           # LLM integration
        ├── models/        # Data models
        ├── retrieval/     # Document retrieval service
        ├── storage/       # Storage interfaces
        └── vector/        # Vector database interfaces
```

## Future Enhancements

- Implement concrete vector database backend (Qdrant, pgvector, etc.)
- Add specialized financial data processors
- Create a web UI for the chat interface
- Add streaming response support
- Support for multi-modal inputs (charts, tables)
