#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelMetadataDemo");
    Ort::SessionOptions session_options;

    // Load your model (replace with actual path)
    Ort::Session session(env, "/assets/models/mnist.onnx", session_options);

    // Get metadata
    Ort::ModelMetadata metadata = session.GetModelMetadata();

    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "--- Model Metadata ---" << std::endl;
    std::cout << "Graph name      : " << metadata.GetGraphNameAllocated(allocator).get() << std::endl;
    std::cout << "Domain          : " << metadata.GetDomainAllocated(allocator).get() << std::endl;
    std::cout << "Description     : " << metadata.GetDescriptionAllocated(allocator).get() << std::endl;
    std::cout << "Producer name   : " << metadata.GetProducerNameAllocated(allocator).get() << std::endl;
    std::cout << "Graph version   : " << metadata.GetVersion() << std::endl;

    // Custom metadata
    auto keys = metadata.GetCustomMetadataMapKeysAllocated(allocator);
    for (auto& key : keys) {
        auto value = metadata.LookupCustomMetadataMapAllocated(key.get(), allocator);
        std::cout << "Custom [" << key.get() << "] = " << value.get() << std::endl;
    }

    return 0;
}
