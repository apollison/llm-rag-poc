#!/bin/bash
set -e

# Clone go-llama.cpp as a submodule in the vendor directory
mkdir -p vendor
cd vendor

# Check if go-llama.cpp is already cloned
if [ ! -d "go-llama.cpp" ]; then
    echo "Cloning go-llama.cpp repository..."
    git clone --recurse-submodules https://github.com/go-skynet/go-llama.cpp
else
    echo "go-llama.cpp already exists, updating..."
    cd go-llama.cpp
    git pull
    git submodule update --init --recursive
    cd ..
fi

# Build the bindings
cd go-llama.cpp
echo "Building go-llama.cpp bindings..."
make libbinding.a

# Try to build with Metal support for Mac
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS, trying to build with Metal support..."
    BUILD_TYPE=metal make libbinding.a
    # Copy the Metal shader file to our project root
    cp build/bin/ggml-metal.metal ../../
fi

cd ../..
echo "Setup complete!"
