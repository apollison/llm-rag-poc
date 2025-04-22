#!/bin/bash
set -e

ROOT_DIR=$(cd ../../../; pwd)
GO_LLAMA_DIR="$ROOT_DIR/vendor/go-llama.cpp"
LLAMA_CPP_DIR="$GO_LLAMA_DIR/llama.cpp"

# Ensure llama.cpp is compiled
if [ ! -f "$GO_LLAMA_DIR/build/libllama.a" ]; then
    echo "Building llama.cpp library..."
    cd "$LLAMA_CPP_DIR"
    mkdir -p build
    cd build
    cmake ..
    cmake --build . --config Release
    # Copy the compiled library to our vendor directory
    cp libllama.a "$GO_LLAMA_DIR/build/"
    cd "$ROOT_DIR/pkg/llm/binding"
fi

# Compile the binding.cpp file
g++ -std=c++11 -c -o binding.o binding.cpp -I$LLAMA_CPP_DIR

# Create the static library
ar rcs libbinding.a binding.o

# Copy to the root of go-llama.cpp directory
cp libbinding.a $GO_LLAMA_DIR/
cp binding.h $GO_LLAMA_DIR/

echo "Built binding library successfully"
