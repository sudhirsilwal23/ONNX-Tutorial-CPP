#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
        return 0;
    }

    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        std::cout << "--- CUDA Device " << device << " ---" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "Total Global Memory: " 
                  << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) 
                  << " GB" << std::endl;

        // Switch to this device
        cudaSetDevice(device);

        // Query memory usage
        size_t free_bytes, total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);

        std::cout << "Total memory: " << (total_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Free memory : " << (free_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "Used memory : " << ((total_bytes - free_bytes) / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
