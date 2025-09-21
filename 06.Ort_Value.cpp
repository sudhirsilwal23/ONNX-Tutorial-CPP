/*

Why do we need Ort:;Value?
-----------------------------------
We need Ort::Value because it's the fundamental data structure used by ONNX Runtime to represent all inputs, outputs, and intermediate data within a model. Think of it as a smart container that holds the raw tensor data along with essential metadata like its shape and data type.

1. Encapsulates Tensor Data
Ort::Value is a high-level wrapper around the raw numerical data. Instead of passing around simple float* or double* pointers, Ort::Value bundles the data with its shape, data type (e.g., float, int64_t), and memory location (e.g., CPU or GPU). This unified structure ensures that the ONNX Runtime engine can correctly interpret and process the data regardless of its source. The provided code creates a tensor and gives it a shape of [1, 3, 2, 2], which tells the runtime how to interpret the one-dimensional input_data vector as a 4D tensor.

2. Required for Model Inference 
The Ort::Session::Run() method, which executes the model, requires its inputs and produces its outputs as Ort::Value objects. You can't pass raw data pointers directly to the Run method. You must first wrap your input data into an Ort::Value object, and the outputs returned by the method will also be Ort::Value objects. This consistent interface is vital for the runtime's internal operations and ensures that the model can be run correctly.

3. Supports Diverse Data Types and Structures 
While the most common use of Ort::Value is for tensors, it's a versatile class that can also represent other data structures supported by the ONNX standard, such as sequences (lists) and maps. This flexibility allows ONNX Runtime to support more complex models that use these data types beyond simple tensors.

*/

#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- ORT::Value Simulation ---" << std::endl;

    // Step 1: Create environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ValueDemo");

    // Step 2: Create allocator
    Ort::AllocatorWithDefaultOptions allocator;

    // Step 3: Define tensor shape (e.g., batch=1, channels=3, height=2, width=2)
    std::vector<int64_t> input_shape = {1, 3, 2, 2};
    size_t input_tensor_size = 1 * 3 * 2 * 2;  // product of shape

    // Step 4: Create some dummy input data
    std::vector<float> input_data(input_tensor_size);
    for (size_t i = 0; i < input_tensor_size; i++) {
        input_data[i] = static_cast<float>(i) / 10.0f;  // fill with 0.0, 0.1, 0.2, ...
    }

    // Step 5: Create Ort::Value tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_tensor_size,
        input_shape.data(), input_shape.size());

    // Step 6: Inspect tensor
    if (input_tensor.IsTensor()) {
        auto type_info = input_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = type_info.GetShape();

        std::cout << "Created Ort::Value as a tensor." << std::endl;
        std::cout << "Shape: [";
        for (size_t i = 0; i < shape.size(); i++) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Number of elements: " << type_info.GetElementCount() << std::endl;
        std::cout << "First element: " << input_data[0] << ", Last element: "
                  << input_data.back() << std::endl;
    } else {
        std::cerr << "Ort::Value is not a tensor!" << std::endl;
    }

    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}
