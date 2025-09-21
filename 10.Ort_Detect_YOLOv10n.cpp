/*

This C++ code performs object detection using a YOLOv10 model through the ONNX Runtime library. The code loads an ONNX model, pre-processes an image to match the model's input requirements, runs inference, and then post-processes the output to draw bounding boxes and labels on the image. The final image with detections is saved to a file.

1. ONNX as an Interchange Format 
The YOLOv10 model is saved in the .onnx format. This is a standard, open format that allows models to be trained in one framework (like PyTorch or TensorFlow) and deployed in another, without being tied to the original framework. The provided code takes advantage of this by using a YOLOv10 model, which was likely trained in a different environment, and running it with the ONNX Runtime engine. This ensures interoperability.

2. ONNX Runtime for High-Performance Inference 
The ONNX Runtime library is the high-performance inference engine that executes the ONNX model. Its primary roles in this code are:

    Model Loading: It reads and parses the .onnx file, creating an in-memory representation of the model's computational graph.
    Graph Optimization: As seen with session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL), the runtime automatically performs aggressive optimizations on the model's graph. This makes the model run faster and more efficiently on the target hardware (in this case, the CPU).
    Inference Execution: The session.Run() method is the core of the process. It takes the prepared input tensor and executes the model's forward pass, performing all the necessary mathematical operations to produce the output tensor containing the object detection results.
    Hardware Abstraction: ONNX Runtime handles the low-level details of running the model on the available hardware. While this code uses the default CPU execution provider, ONNX Runtime could seamlessly switch to a GPU by appending the CUDA execution provider, without changing the core model-running logic.

*/


#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// ---------------- Main ----------------
int main() {
    try {
        std::cout << "--- YOLOv10 ONNX Inference Demo (Post-Processed Output) ---" << std::endl;

        // 1. ORT Environment + Session
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv10Demo");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        const std::string model_path = "/assets/models/yolov10n.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);

        // 2. Load input image
        const std::string image_path = "/assets/images/car.png";
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "❌ Error: could not load image at " << image_path << std::endl;
            return -1;
        }

        int input_w = 640, input_h = 640;
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(input_w, input_h));
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);

        // 3. Prepare input tensor (HWC -> CHW)
        std::vector<int64_t> input_shape = {1, 3, input_h, input_w};
        std::vector<float> input_tensor_values;
        std::vector<cv::Mat> channels(3);
        cv::split(resized, channels);

        for (int c = 0; c < 3; c++) {
            input_tensor_values.insert(input_tensor_values.end(),
                                       (float*)channels[c].datastart,
                                       (float*)channels[c].dataend);
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // 4. Fetch input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = {input_name_alloc.get()};
        const char* output_names[] = {output_name_alloc.get()};

        // 5. Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        // 6. Parse output (already [N,6]: x1,y1,x2,y2,conf,class_id)
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        auto output_shape = type_info.GetShape();

        int num_dets = output_shape[1]; // e.g., 300
        int num_attrs = output_shape[2]; // should be 6

        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shape.size(); i++) {
            std::cout << output_shape[i] << (i < output_shape.size() - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;

        for (int i = 0; i < num_dets; i++) {
            float x1   = output_data[i * num_attrs + 0];
            float y1   = output_data[i * num_attrs + 1];
            float x2   = output_data[i * num_attrs + 2];
            float y2   = output_data[i * num_attrs + 3];
            float conf = output_data[i * num_attrs + 4];
            int class_id = static_cast<int>(output_data[i * num_attrs + 5]);

            if (conf < 0.25f) continue; // confidence threshold

            cv::Rect box((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
            cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);

            cv::putText(image,
                        "cls " + std::to_string(class_id) + ":" + cv::format("%.2f", conf),
                        cv::Point(x1, y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 0, 0), 1);
        }

        // 7. Save result
        const std::string output_path = "/assets/output/yolov10_car_output.jpg";
        cv::imwrite(output_path, image);
        std::cout << "✅ Detection complete. Saved as " << output_path << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }
    catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
