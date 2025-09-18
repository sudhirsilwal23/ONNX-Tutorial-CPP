#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- ORT::Session::Run with MNIST ---" << std::endl;

    // Step 1: Create environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MNISTDemo");

    // Step 2: Create session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Step 3: Load the MNIST ONNX model
    const char* model_path = "mnist.onnx"; // <-- make sure this file exists
    Ort::Session session(env, model_path, session_options);

    // Step 4: Inspect input info
    Ort::AllocatorWithDefaultOptions allocator;
    char* input_name = session.GetInputName(0, allocator);
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType input_type = input_tensor_info.GetElementType();
    std::vector<int64_t> input_shape = input_tensor_info.GetShape();

    std::cout << "Model input name: " << input_name << "\n";
    std::cout << "Input shape: [";
    for (size_t i = 0; i < input_shape.size(); i++) {
        std::cout << input_shape[i];
        if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Step 5: Prepare dummy input (MNIST expects [1,1,28,28])
    size_t input_tensor_size = 1 * 1 * 28 * 28;
    std::vector<float> input_data(input_tensor_size, 0.0f);
    input_data[0] = 1.0f; // just a simple dummy pixel

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_tensor_size,
        input_shape.data(),
        input_shape.size());

    // Step 6: Prepare output
    char* output_name = session.GetOutputName(0, allocator);

    // Step 7: Run inference
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,   // inputs
        &output_name, 1                  // outputs
    );

    // Step 8: Extract results
    float* output_arr = output_tensors.front().GetTensorMutableData<float>();
    auto output_type_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_type_info.GetShape();

    std::cout << "Output shape: [";
    for (size_t i = 0; i < output_shape.size(); i++) {
        std::cout << output_shape[i];
        if (i < output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // MNIST outputs 10 class probabilities
    std::cout << "Output probabilities: ";
    for (int i = 0; i < 10; i++) {
        std::cout << output_arr[i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << "\n";

    // Step 9: Get predicted label
    int predicted = std::distance(output_arr, std::max_element(output_arr, output_arr + 10));
    std::cout << "Predicted digit: " << predicted << "\n";

    allocator.Free(input_name);
    allocator.Free(output_name);

    std::cout << "--- Inference Complete ---" << std::endl;
    return 0;
}
