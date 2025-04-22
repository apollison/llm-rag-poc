#ifndef BINDING_H
#define BINDING_H

// Use a relative path to the llama.cpp directory
#include "../../../vendor/go-llama.cpp/llama.cpp/llama.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_context_params llama_context_default_params_wrapper();
struct llama_model* llama_load_model_from_file_wrapper(const char* path_model, struct llama_context_params* params);
struct llama_context* llama_new_context_with_model_wrapper(struct llama_model* model, struct llama_context_params* params);
const char* llama_predict(void* ctx_ptr, void* model_ptr, const char* prompt, int32_t seed,
                         int32_t threads, int32_t tokens, float temp, float top_p,
                         int32_t top_k, float repeat_penalty);

#ifdef __cplusplus
}
#endif

#endif // BINDING_H
