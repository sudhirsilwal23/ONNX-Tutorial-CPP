/*

Why do we need Ort:Allocator?
----------------------------------

You need an Ort::Allocator to manage memory for ONNX Runtime operations, ensuring data is allocated and freed correctly, especially
when working with different hardware like CPUs and GPUs.

It provides a standardized way for the runtime to handle memory, which is essential for creating inputs and outputs that can be
processed by a model.

*/

#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- Ort::Allocator Simulation ---" << std::endl;

    // Step 1: Create allocator (default CPU allocator)
    Ort::AllocatorWithDefaultOptions allocator;

    // Step 2: Allocate raw memory for 10 floats
    size_t num_elements = 10;
    size_t size_in_bytes = num_elements * sizeof(float);

    void* raw_memory = allocator.Alloc(size_in_bytes);

    if (!raw_memory) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return -1;
    }

    std::cout << "Allocated " << size_in_bytes 
              << " bytes using Ort::AllocatorWithDefaultOptions." << std::endl;

    // Step 3: Fill memory with sample values
    float* data_ptr = reinterpret_cast<float*>(raw_memory);
    for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = static_cast<float>(i) * 0.1f;
    }

    // Step 4: Create tensor using allocator memory
    std::vector<int64_t> shape = {static_cast<int64_t>(num_elements)};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem_info, data_ptr, num_elements, shape.data(), shape.size()
    );

    if (tensor.IsTensor()) {
        auto type_info = tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> tensor_shape = type_info.GetShape();

        std::cout << "Tensor created successfully! Shape: [";
        for (size_t i = 0; i < tensor_shape.size(); i++) {
            std::cout << tensor_shape[i];
            if (i < tensor_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Tensor values: ";
        float* tensor_data = tensor.GetTensorMutableData<float>();
        for (size_t i = 0; i < num_elements; i++) {
            std::cout << tensor_data[i] << " ";
        }
        std::cout << std::endl;
    }

    // Step 5: Free the allocated memory
    allocator.Free(raw_memory);
    std::cout << "Memory freed successfully." << std::endl;

    std::cout << "--- Simulation Complete ---" << std::endl;
    return 0;
}
