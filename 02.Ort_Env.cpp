/*
In ONNX Runtime C++ API, Ort::Env is one of the very first objects you create.
It represents the global environment for the ONNX Runtime process and is responsible for:

Purpose of Ort::Env
-------------------------------------------
- Initialization: It initializes the ONNX Runtime library (thread pools, logging, etc.).
- Logging: You can configure the logging level and the log identifier.
- Lifetime Management: You should usually have only one Ort::Env per process. Creating multiple is discouraged because it leads to multiple thread pools and duplicated resources.
- Parent for Sessions: Any Ort::Session object (your loaded ONNX model) is tied to this environment.


In C++ API it is mandatory to create an Ort::Env before you create any session (Ort::Session).

Here’s why:
---------------------------------------------
Ort::Env initializes the global runtime state (thread pools, memory allocators, logging).
Every Ort::Session (model instance) requires an existing Ort::Env to be passed in its constructor.

-------------------------------------------
You cannot load a model without Ort::Env.
-------------------------------------------
*/


#include <iostream>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "--- ORT Env Simulation ---" << std::endl;

    // Create an ORT Environment
    // First argument = logging level
    // Second argument = log identifier (useful for debugging multiple sessions)
    Ort::Env env(ORT_LOGGING_LEVEL_INFO, "EnvSimulation");

    std::cout << "Environment created successfully!" << std::endl;

    // You can create session options to pass with this environment
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    std::cout << "SessionOptions created. IntraOp threads set to 1." << std::endl;

    // Example: enabling optimizations
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    std::cout << "Graph optimization set to BASIC." << std::endl;

    // The environment will stay alive until program exit
    std::cout << "Simulation complete. (No model loaded here)." << std::endl;

    return 0;
}
