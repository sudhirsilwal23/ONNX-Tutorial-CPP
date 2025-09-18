/*

Ort::MemoryInfo in ONNX Runtime C++ API, shows how you can query and print properties of 
memory allocators that ONNX Runtime uses when handling tensors.

This is useful when you:
-------------------------------------------
- Create tensors with custom memory (you want ONNX Runtime to wrap your buffer).
- Need to move data between CPU and GPU correctly.
- Debug which device your tensors are being allocated on.

*/


#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- Ort::MemoryInfo Properties Demo ---\n";

    // 1️⃣ CPU MemoryInfo (4-arg constructor)
    Ort::MemoryInfo cpu_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
                                
                                // "Cpu"              → allocator name.
                                // OrtDeviceAllocator → default allocator type.
                                // 0                  → device ID (CPU has no device ID, but 0 is used).
                                // OrtMemTypeDefault  → normal memory.
                                

    std::cout << "[CPU MemoryInfo]\n";
    std::cout << "  Allocator Name : " << cpu_info.GetAllocatorName() << "\n";
    std::cout << "  Allocator Type : " << cpu_info.GetAllocatorType() << "\n";
    std::cout << "  Device Type    : " << cpu_info.GetDeviceType() << "\n";
    std::cout << "  Device ID      : " << cpu_info.GetDeviceId() << "\n";
    std::cout << "  Memory Type    : " << cpu_info.GetMemoryType() << "\n\n";

    // 2️⃣ CUDA MemoryInfo (if available)
    try {
        Ort::MemoryInfo cuda_info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);

        std::cout << "[CUDA MemoryInfo]\n";
        std::cout << "  Allocator Name : " << cuda_info.GetAllocatorName() << "\n";
        std::cout << "  Allocator Type : " << cuda_info.GetAllocatorType() << "\n";
        std::cout << "  Device Type    : " << cuda_info.GetDeviceType() << "\n";
        std::cout << "  Device ID      : " << cuda_info.GetDeviceId() << "\n";
        std::cout << "  Memory Type    : " << cuda_info.GetMemoryType() << "\n\n";
    } catch (const Ort::Exception& e) {
        std::cerr << "CUDA MemoryInfo not available: " << e.what() << "\n";
    }

    std::cout << "--- Demo Complete ---\n";
    return 0;
}
