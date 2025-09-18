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
