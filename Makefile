.PHONY: setup build build-chat build-rag build-indexer run-chat clean copy-headers

VENDOR_DIR := $(PWD)/vendor/go-llama.cpp
LLAMA_CPP_DIR := $(VENDOR_DIR)/llama.cpp
COMMON_DIR := $(LLAMA_CPP_DIR)/common

# More comprehensive include paths for CGO
CGO_CFLAGS := -I$(VENDOR_DIR) -I$(LLAMA_CPP_DIR) -I$(COMMON_DIR)
CGO_LDFLAGS := -L$(VENDOR_DIR) -Wl,-rpath,$(VENDOR_DIR)
GOLLAMA_TAG := cgo

# Check if we're on macOS for Metal support
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	METAL_SHADER := ggml-metal.metal
	CGO_LDFLAGS += -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
endif

all: setup build

setup:
	@echo "Setting up go-llama.cpp..."
	./setup-llama.sh
	@echo "Setup complete"

# Copy required header files directly to the vendor/go-llama.cpp directory
copy-headers:
	@echo "Copying header files..."
	cp -n $(COMMON_DIR)/common.h $(VENDOR_DIR)/ || true
	cp -n $(COMMON_DIR)/grammar-parser.h $(VENDOR_DIR)/ || true
	cp -n $(LLAMA_CPP_DIR)/ggml.h $(VENDOR_DIR)/ || true
	cp -n $(LLAMA_CPP_DIR)/llama.h $(VENDOR_DIR)/ || true
	@echo "Copied header files"

build: copy-headers build-chat build-rag build-indexer

build-chat: copy-headers
	@echo "Building chat binary..."
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags $(GOLLAMA_TAG) -o bin/chat ./cmd/chat

build-rag: copy-headers
	@echo "Building RAG service binary..."
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags $(GOLLAMA_TAG) -o bin/rag-service ./cmd/rag-service

build-indexer: copy-headers
	@echo "Building data indexer binary..."
	CGO_CFLAGS="$(CGO_CFLAGS)" CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags $(GOLLAMA_TAG) -o bin/data-indexer ./cmd/data-indexer

run-chat: build-chat
	bin/chat

clean:
	@echo "Cleaning generated files..."
	rm -rf bin/
	@echo "Clean complete"
