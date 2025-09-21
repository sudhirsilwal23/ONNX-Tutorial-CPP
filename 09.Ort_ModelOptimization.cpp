/*
We need ONNX Runtime model optimization to improve the inference performance of a model. It's a crucial step that transforms a standard 
ONNX model into a version that runs faster and more efficiently on a target device.

The optimization process is essentially a series of graph-level transformations that make the model more efficient without affecting 
its accuracy.  These optimizations are applied when the model is loaded into an Ort::Session by setting the SetGraphOptimizationLevel 
on the Ort::SessionOptions
*/


#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- ONNX Model Optimization Simulation ---" << std::endl;

    // Step 1: Create environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelOptimizationDemo");

    // Step 2: Create session options
    Ort::SessionOptions session_options;

    // 🔹 Enable graph optimization
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);


     // (Optional) Save optimized model to disk
    // Uncomment to save optimized model for inspection
    // session_options.SetOptimizedModelFilePath("mnist_optimized.onnx");

    // Enable CPU memory arena and threading
    session_options.EnableCpuMemArena();
    session_options.SetIntraOpNumThreads(2);

    // Step 3: Load a model (replace with your actual ONNX path)
    const char* model_path = "/assets/models/mnist.onnx";
    std::cout << "Loading model: " << model_path << std::endl;

    try {
        Ort::Session session(env, model_path, session_options);

        // Step 4: Query model metadata
        Ort::ModelMetadata metadata = session.GetModelMetadata();
        Ort::AllocatorWithDefaultOptions allocator;

        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Graph optimizations applied: ORT_ENABLE_ALL" << std::endl;

        // ✅ Safe custom metadata lookup
        try {
            auto value = metadata.LookupCustomMetadataMapAllocated("model_version", allocator);
            if (value) {
                std::cout << "Model version: " << value.get() << std::endl;
            }
        } catch (const Ort::Exception& e) {
            std::cout << "No custom metadata field 'model_version' found." << std::endl;
        }

        // Step 5: Print input/output info
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "Inputs: " << num_input_nodes << ", Outputs: " << num_output_nodes << std::endl;

        for (size_t i = 0; i < num_input_nodes; i++) {
            auto name = session.GetInputNameAllocated(i, allocator);
            std::cout << "  Input " << i << " : " << name.get() << std::endl;
        }
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            std::cout << "  Output " << i << " : " << name.get() << std::endl;
        }
    }
    catch (const Ort::Exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}
