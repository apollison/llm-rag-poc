package binding

// #cgo CXXFLAGS: -std=c++11 -I${SRCDIR} -I${SRCDIR}/../../../vendor/go-llama.cpp/llama.cpp
// #cgo LDFLAGS: -L${SRCDIR} -L${SRCDIR}/../../../vendor/go-llama.cpp -L${SRCDIR}/../../../vendor/go-llama.cpp/build -L${SRCDIR}/../../../vendor/go-llama.cpp/llama.cpp/build -lbinding -lllama -lstdc++ -lm
// #cgo darwin LDFLAGS: -framework Accelerate
// #include <stdlib.h>
// #include "binding.h"
import "C"
import (
	"errors"
	"unsafe"
)

// LlamaModel represents a llama model
type LlamaModel struct {
	modelPtr unsafe.Pointer
	ctxPtr   unsafe.Pointer
}

// NewLlamaModel creates a new llama model
func NewLlamaModel(modelPath string, contextSize, seed, threads int) (*LlamaModel, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	// Get default params
	params := C.llama_context_default_params_wrapper()

	// Set parameters using the correct field names from llama_context_params
	params.n_ctx = C.int32_t(contextSize)
	params.seed = C.uint32_t(seed)

	// Note: There's no direct threads parameter in llama_context_params
	// We could adjust n_batch which might roughly correspond to our desired threads setting
	params.n_batch = C.int32_t(threads)

	// Load model
	model := C.llama_load_model_from_file_wrapper(cModelPath, &params)
	if model == nil {
		return nil, errors.New("failed to load model")
	}

	// Create context
	ctx := C.llama_new_context_with_model_wrapper(model, &params)
	if ctx == nil {
		C.llama_free_model(model)
		return nil, errors.New("failed to create context")
	}

	return &LlamaModel{
		modelPtr: unsafe.Pointer(model),
		ctxPtr:   unsafe.Pointer(ctx),
	}, nil
}

// Predict generates text from the model
func (m *LlamaModel) Predict(prompt string, tokens, seed, threads int, temp, topP float32, topK int, repeatPenalty float32) (string, error) {
	if m.modelPtr == nil || m.ctxPtr == nil {
		return "", errors.New("model not initialized")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	result := C.llama_predict(
		m.ctxPtr,
		m.modelPtr,
		cPrompt,
		C.int32_t(seed),
		C.int32_t(threads),
		C.int32_t(tokens),
		C.float(temp),
		C.float(topP),
		C.int32_t(topK),
		C.float(repeatPenalty),
	)

	if result == nil {
		return "", errors.New("failed to generate prediction")
	}

	return C.GoString(result), nil
}

// Free releases resources used by the model
func (m *LlamaModel) Free() {
	if m.ctxPtr != nil {
		C.llama_free((*C.struct_llama_context)(m.ctxPtr))
		m.ctxPtr = nil
	}
	if m.modelPtr != nil {
		C.llama_free_model((*C.struct_llama_model)(m.modelPtr))
		m.modelPtr = nil
	}
}
