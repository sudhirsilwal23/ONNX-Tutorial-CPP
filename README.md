# 🚀 ONNX-Tutorial-CPP

**ONNX-Tutorial-C++** is a hands-on guide for learning how to integrate and use **ONNX Runtime** in modern **C++ applications**.  
This repository is designed for **AI engineers, researchers, and C++ developers** who want to run deep learning inference with ONNX Runtime in C++ using practical examples.  

---

## 📂 Repository Contents  

Each `.cpp` file demonstrates a **specific ONNX Runtime concept** or **object detection workflow**.  

| File | Concept | Description |
|------|---------|-------------|
| `1.Ort_Env.cpp` | `Ort::Env` | Shows how to create and configure the ONNX Runtime environment. |
| `2.Ort_SessionOptions.cpp` | `Ort::SessionOptions` | Demonstrates threading, optimization levels, and session settings. |
| `3.Ort_MemoryInfo.cpp` | `Ort::MemoryInfo` | Explains CPU/GPU memory allocation strategies. |
| `4.Ort_Session.cpp` | `Ort::Session` | Loading ONNX models into an ORT session. |
| `5.Ort_Value.cpp` | `Ort::Value` | Creating and working with tensors in C++. |
| `6.Ort_Run.cpp` | `Ort::Session::Run` | Running inference with dummy inputs/outputs. |
| `7.Ort_Allocator.cpp` | `Ort::Allocator` | How ONNX Runtime allocates memory for tensors. |
| `8.ORT_ModelOptimization.cpp` | Optimization | Simulating graph optimizations with ORT. |
| `9.Ort_Detect_YOLOv10n.cpp` | Object Detection | Running **YOLOv10n ONNX** for inference using OpenCV + ORT. |
| `assets/models/` | Models | Place your `.onnx` models here (e.g., YOLOv10n). |
| `assets/images/` | Images | Test images (e.g., `car.jpg`). |

---

## 🛠️ Setup Instructions  

### 1️⃣ Install ONNX Runtime C++  

Download ONNX Runtime prebuilt binaries:  
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
tar -xvzf onnxruntime-linux-x64-gpu-1.18.1.tgz

