/*

Why do wee need Ort::Session?
----------------------------------------------------
We need Ort::Session because it is the central object that encapsulates and manages a loaded ONNX model. It's the primary interface for running inference and interacting with the model.
Think of it as the engine that drives all model-related operations.

1. Model Loading and Optimization 
The Ort::Session constructor is responsible for loading the ONNX model file from disk. During this process, it applies all the optimizations and configurations specified by the Ort::SessionOptions object. 
This includes setting the number of threads for parallelism (SetIntraOpNumThreads) and applying graph-level optimizations (SetGraphOptimizationLevel).
Without a session, these optimizations would never be applied, and the model would not be prepared for efficient execution.

2. Abstraction of Model Details 
The session object provides methods to query important information about the model, such as the number of inputs and outputs (GetInputCount, GetOutputCount), 
their names (GetInputNameAllocated, GetOutputNameAllocated), and their shapes and data types (GetInputTypeInfo, GetOutputTypeInfo). This is crucial for programmatically preparing
the correct input tensors and interpreting the output results. It abstracts away the complex internal structure of the ONNX file, presenting a clean and accessible interface.

3. Inference Execution 
Ultimately, the session is the object you call to run the actual inference on the model. The session.Run() method takes the prepared input tensors and executes 
the model's forward pass, producing the output tensors. The Ort::Session manages the entire computational pipeline, from data flow to kernel execution on the chosen hardware (CPU, GPU, etc.), making it an indispensable part of the ONNX Runtime API.

*/



#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

// Utility: print shape
void printShape(const std::vector<int64_t>& shape) {
    std::cout << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i != shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]";
}

int main() {
    std::cout << "--- ORT Session Simulation ---" << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SessionDemo");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(2);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    const char* model_path = "/assets/models/mnist.onnx";

    try {
        Ort::Session session(env, model_path, session_options);
        std::cout << "Session created successfully for model: " << model_path << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;

        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();

        std::cout << "Model has " << num_inputs << " input(s) and "
                  << num_outputs << " output(s)." << std::endl;

        for (size_t i = 0; i < num_inputs; i++) {
            Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
            Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            std::vector<int64_t> input_shape = tensor_info.GetShape();
            std::cout << "Input " << i << " name: " << input_name.get() << " | Shape: ";
            printShape(input_shape);
            std::cout << std::endl;
        }

        for (size_t i = 0; i < num_outputs; i++) {
            Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
            Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            std::vector<int64_t> output_shape = tensor_info.GetShape();
            std::cout << "Output " << i << " name: " << output_name.get() << " | Shape: ";
            printShape(output_shape);
            std::cout << std::endl;
        }

        std::cout << "Inference would be run with session.Run(...) if inputs were provided." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to create session: " << e.what() << std::endl;
    }

    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}
