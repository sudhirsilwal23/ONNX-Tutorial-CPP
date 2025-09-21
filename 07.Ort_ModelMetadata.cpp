/*
Why do we need Ort::ModelMetadata?
-------------------------------------------------

We need Ort::ModelMetadata to access and inspect non-tensor-related information about an ONNX model. While Ort::Session gives us details about a model's inputs and outputs, Ort::ModelMetadata provides a higher-level view, including descriptive and administrative details.

1. Model Identification 
Ort::ModelMetadata allows you to retrieve basic information that helps identify and categorize a model. This includes its graph name, domain, and producer name. This is essential for managing large numbers of models, ensuring you're using the correct one for a task, and for debugging purposes.

2. Versioning and Documentation 
The metadata object provides access to the model's version and a descriptive text. This information is critical for version control and documenting a model's purpose, usage, or any specific constraints. Developers can use this to programmatically check for version compatibility or to display helpful information to end-users.

3. Custom Metadata 
One of the most powerful features is the ability to read custom metadata. ONNX allows model creators to embed key-value pairs of their own choosing into the model file. Ort::ModelMetadata provides methods to retrieve these keys and their corresponding values. This can be used for:

- Storing training parameters (e.g., learning rate, number of epochs).
- Defining model-specific pre-processing or post-processing instructions.
- Tracking the model's lineage or its source repository.

This makes the ONNX model a self-contained unit of information, not just a set of weights and operations.

*/


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
