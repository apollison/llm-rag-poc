# llm-rag-poc

RAG (Retrieval-Augmented Generation) system for financial data and trading strategies. Focus on stock and options trading.

## Overview

This project implements a RAG system that helps answer questions about stocks, options, and trading strategies using a combination of:

- Vector embeddings (via Ollama)
- Vector database storage (via Qdrant)
- LLM-based question answering (via Ollama)

The system consists of two main components:
1. A content indexer that processes documents into vectors
2. A query tool that answers questions using RAG

## Prerequisites

- Go 1.24+
- [Ollama](https://ollama.ai/) running locally with:
  - `llama3.2` model for chat
  - `snowflake-arctic-embed:33m` model for embeddings
- [Qdrant](https://qdrant.tech/) vector database

## Setup

### Install Dependencies

```bash
go get github.com/ollama/ollama/api
go get github.com/qdrant/go-client
go get google.golang.org/grpc
```

### Running Qdrant

The easiest way to run Qdrant is using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

## Usage

### 1. Index Content

First, process your content files to create vectors and store them in Qdrant:

```bash
cd cmd/rag-indexer
go build
./rag-indexer --debug
```

Options:
- `--debug`: Enable verbose logging
- `--qdrant-host`: Specify Qdrant host (default: localhost)
- `--qdrant-port`: Specify Qdrant gRPC port (default: 6334)
- `--content-dir`: Specify content directory (default: ../../content)
- `--recreate`: Recreate collection if it exists

### 2. Query the System

After indexing, you can query the system:

```bash
cd cmd/ollama-rag
go build
./ollama-rag                   # Single question mode
./ollama-rag --interactive     # Interactive chat mode
```

Options:
- `--debug`: Enable verbose logging
- `--interactive`: Run in interactive chat mode
- `--qdrant-host`: Specify Qdrant host (default: localhost)
- `--qdrant-port`: Specify Qdrant gRPC port (default: 6334)

## Content Structure

Place your content files in the `content/` directory. The system automatically processes all `.md` files:

- `content/market_data/`: Market data like option chains
- `content/portfolio/`: Portfolio positions
- `content/strategies/`: Trading strategies and rules

## Example Usage

Single question mode:
```bash
./ollama-rag
```

Interactive chat mode:
```bash
./ollama-rag --interactive
```

Debug mode with custom Qdrant location:
```bash
./ollama-rag --interactive --debug --qdrant-host=192.168.1.100 --qdrant-port=6334
```

Re-index content with a clean database:
```bash
./rag-indexer --recreate --debug
```
