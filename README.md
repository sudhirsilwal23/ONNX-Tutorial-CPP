# 🚀 ONNX-Tutorial-CPP

**ONNX-Tutorial-C++** is a hands-on guide for learning how to integrate and use **ONNX Runtime** in modern **C++ applications**.  
This repository is designed for **AI engineers, researchers, and C++ developers** who want to run deep learning inference with ONNX Runtime in C++ using practical examples.  

---

## 📂 Repository Contents  

### 🔑 Core ONNX Runtime Concepts
| File | Concept | Description |
|------|---------|-------------|
| `1.Ort_MemoryInfo.cpp` | `Ort::MemoryInfo` | Demonstrates CPU/GPU memory allocation strategies. |
| `2.Ort_Env.cpp` | `Ort::Env` | Creates and configures the ONNX Runtime environment. |
| `3.Ort_SessionOptions.cpp` | `Ort::SessionOptions` | Configures threading, graph optimizations, and profiling. |
| `4.Ort_Allocator.cpp` | `Ort::Allocator` | Shows how ONNX Runtime allocates memory for tensors. |
| `5.Ort_Session.cpp` | `Ort::Session` | Loads an ONNX model into a session. |
| `6.Ort_Value.cpp` | `Ort::Value` | Creating and working with tensors in C++. |
| `7.Ort_ModelMetadata.cpp` | Model Metadata | Reads model name, domain, version, and metadata. |
| `8.ORT_Session_Run.cpp` | `Ort::Session::Run` | Running inference with inputs/outputs. |
| `9.Ort_ModelOptimization.cpp` | Optimization | Simulating graph optimizations with ORT. |

### 🎯 Object Detection Examples
| File | Concept | Description |
|------|---------|-------------|
| `10.Ort_Detect_YOLOv10n.cpp` | YOLOv10n Detection | End-to-end object detection with OpenCV + ONNX Runtime. |

### 🎯 CUDA Examples
| File | Concept | Description |
|------|---------|-------------|
| `11.cuda_memory_info.cpp` | CUDA Memory | Querying CUDA device memory info (free/total memory). |

### 📂 Assets
- `assets/models/` → Place ONNX models here (e.g., `yolov10n.onnx`).  
- `assets/images/` → Place test images here (e.g., `car.jpg`, `lena.png`).  


---

## 🛠️ Setup Instructions  

### 1️⃣ Install ONNX Runtime C++  


Download the pre-built ONNX Runtime binaries. This example uses the **GPU version** for Linux.

```bash
2️⃣ Export Path
export ONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-1.18.1

✅ How to Compile and Run
1. Ort::MemoryInfo Demo

Compile:

g++ -std=c++17 1.Ort_MemoryInfo.cpp \
    -I $ONNXRUNTIME_ROOT/include \
    -L $ONNXRUNTIME_ROOT/lib -lonnxruntime \
    -Wl,-rpath,$ONNXRUNTIME_ROOT/lib \
    -o ort_memory_info


Run:

./ort_memory_info

2. YOLOv10n Object Detection

Compile:

g++ -std=c++17 10.Ort_Detect_YOLOv10n.cpp \
    -I $ONNXRUNTIME_ROOT/include \
    -L $ONNXRUNTIME_ROOT/lib -lonnxruntime \
    `pkg-config --cflags --libs opencv4` \
    -Wl,-rpath,$ONNXRUNTIME_ROOT/lib \
    -o ort_yolo10n


Run:

LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ./ort_yolo10n

3. CUDA Memory Info

Compile:

g++ -std=c++17 11.cuda_memory_info.cpp \
    -I $ONNXRUNTIME_ROOT/include \
    -L $ONNXRUNTIME_ROOT/lib -lonnxruntime \
    -lcuda -lcudart \
    -Wl,-rpath,$ONNXRUNTIME_ROOT/lib \
    -o cuda_memory_info


Run:

./cuda_memory_info
