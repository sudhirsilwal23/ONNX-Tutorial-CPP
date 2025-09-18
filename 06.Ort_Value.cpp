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
