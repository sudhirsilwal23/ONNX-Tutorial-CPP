#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- ORT SessionOptions Simulation ---" << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SessionOptionsDemo");
    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(2);
    std::cout << "IntraOp threads set to 2." << std::endl;

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    std::cout << "Graph optimization level set to EXTENDED." << std::endl;

    session_options.EnableProfiling("profiling_output.json");
    std::cout << "Profiling enabled." << std::endl;

    session_options.DisableMemPattern();
    std::cout << "Memory pattern optimization disabled." << std::endl;

    // ✅ Define CUDA options
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;  // use GPU 0
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = SIZE_MAX;  // no explicit limit
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;

    // Append CUDA EP with options
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    std::cout << "CUDA Execution Provider appended (device 0)." << std::endl;

    // (No model loaded here)
    std::cout << "Simulation complete." << std::endl;
    return 0;
}
