#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <thread>
#include <iostream>
#include <memory>

#include "llama.h"
#include "binding.h"

// Result storage for text generation
thread_local std::string generation_result;

void llama_binding_initialize() {
    // Initialize llama backend
    llama_backend_init(false);
}

llama_model_handle llama_binding_load_model(const char* path, int n_ctx, uint32_t seed, int n_threads) {
    // Set up model parameters - llama.cpp API has changed to use llama_model_params
    struct llama_model_params model_params = llama_model_default_params();

    // Load the model
    llama_model* model = llama_load_model_from_file(path, model_params);
    if (!model) {
        std::cerr << "Failed to load model from " << path << std::endl;
        return nullptr;
    }

    return static_cast<llama_model_handle>(model);
}

llama_context_handle llama_binding_create_context(llama_model_handle model_handle) {
    if (!model_handle) {
        return nullptr;
    }

    llama_model* model = static_cast<llama_model*>(model_handle);

    // Set up context parameters
    struct llama_context_params ctx_params = llama_context_default_params();

    // Create context
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context" << std::endl;
        return nullptr;
    }

    return static_cast<llama_context_handle>(ctx);
}

const char* llama_binding_generate(llama_context_handle ctx_handle, const char* prompt,
                                int max_tokens, float temperature, float top_p, int top_k) {
    if (!ctx_handle) {
        return nullptr;
    }

    llama_context* ctx = static_cast<llama_context*>(ctx_handle);

    // Tokenize the prompt - updated API for llama_tokenize
    std::vector<llama_token> tokens(1024); // Allocate token buffer
    int n_tokens = llama_tokenize(ctx, prompt, strlen(prompt), tokens.data(), tokens.size(), true);
    if (n_tokens < 0) {
        std::cerr << "Failed to tokenize prompt" << std::endl;
        return nullptr;
    }
    tokens.resize(n_tokens);

    // Create batch for inference
    struct llama_batch batch = {
        n_tokens,            // n_tokens
        tokens.data(),       // token
        nullptr,             // embd
        nullptr,             // pos
        nullptr,             // n_seq_id
        nullptr,             // seq_id
        nullptr,             // logits
        0, 0, 0              // other fields initialized to 0
    };

    // Allocate arrays for pos and logits flags
    std::vector<llama_pos> positions(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        positions[i] = i;
    }
    batch.pos = positions.data();

    std::vector<int32_t> logits(n_tokens, 0);
    logits[n_tokens-1] = 1;  // only need logits for the last token
    batch.logits = logits.data();

    // Process the prompt
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Failed to decode" << std::endl;
        return nullptr;
    }

    // Generate response
    std::stringstream ss;

    for (int i = 0; i < max_tokens; i++) {
        // Get logits for sampling
        float* logits_ptr = llama_get_logits(ctx);
        int n_vocab = llama_n_vocab(ctx);

        // Prepare candidates for sampling
        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates.push_back({ token_id, logits_ptr[token_id], 0.0f });
        }

        llama_token_data_array candidates_array = {
            candidates.data(),
            candidates.size(),
            false
        };

        // Apply sampling parameters
        if (temperature > 0) {
            llama_sample_top_k(ctx, &candidates_array, top_k, 1);
            llama_sample_top_p(ctx, &candidates_array, top_p, 1);
            llama_sample_temp(ctx, &candidates_array, temperature);
        }

        // Sample token
        llama_token token_id = llama_sample_token(ctx, &candidates_array);

        // Check for EOS
        if (token_id == llama_token_eos(ctx)) {
            break;
        }

        // Convert token to string
        const char* piece = llama_token_to_piece(ctx, token_id);
        if (piece) {
            ss << piece;
        }

        // Prepare a batch for a single token
        batch = { 0 };  // Reset batch
        batch.n_tokens = 1;
        batch.token = &token_id;
        batch.pos = &positions[0];
        positions[0] = n_tokens + i;
        batch.logits = &logits[0];
        logits[0] = 1;

        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "Failed to decode" << std::endl;
            break;
        }
    }

    // Store the result in thread_local storage to ensure it remains valid
    generation_result = ss.str();
    return generation_result.c_str();
}

void llama_binding_free_context(llama_context_handle ctx_handle) {
    if (ctx_handle) {
        llama_context* ctx = static_cast<llama_context*>(ctx_handle);
        llama_free(ctx);
    }
}

void llama_binding_free_model(llama_model_handle model_handle) {
    if (model_handle) {
        llama_model* model = static_cast<llama_model*>(model_handle);
        llama_free_model(model);
    }
}
