/*

Why do we need session.Run(...)?
---------------------------------------------------------

We need Ort::Session::Run because it is the function that executes the neural network model. It's the command that tells ONNX Runtime 
to perform the forward pass, taking the prepared input data and producing the final output. It is the core of the inference process.

The Ort::Session::Run method orchestrates the entire inference pipeline, abstracting away the complex details of computation and data flow.
1. Input-to-Output Mapping: The Run method takes three sets of arrays as input:

    1. An array of input names (char* input_name): This tells the runtime which input tensor corresponds to which named input node in the ONNX graph.
    2. An array of input values (Ort::Value* input_tensor): This provides the actual input data wrapped in Ort::Value objects.
    3. An array of output names (char* output_name): This specifies which output nodes the user wants to retrieve results from.

This explicit mapping ensures that data is sent to the correct places in the model, and the desired results are returned.

2. Graph Execution: Once the Run method is called, ONNX Runtime takes control. It uses the Ort::Session object's internal 
representation of the model and its configurations to efficiently execute the computational graph. This includes:

    - Operator Dispatch: Running the correct implementation for each operation (e.g., convolution, ReLU, etc.).
    - Execution Provider Selection: Utilizing the appropriate hardware (CPU, GPU, etc.) based on the session's configuration.
    - Data Flow Management: Moving data between layers and allocating/freeing memory as needed.

3. Result Retrieval: The Run method returns a std::vector<Ort::Value>, where each Ort::Value in the vector corresponds to an output 
specified in the input list of output names. This is the final step where the results of the model's prediction are provided back 
to the user's application for further processing.


*/



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
