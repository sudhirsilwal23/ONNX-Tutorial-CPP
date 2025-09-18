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

wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz
tar -xvzf onnxruntime-linux-x64-gpu-1.18.1.tgz



### ✅ How to Compile and Run

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

### 📌 Key Learnings

✔️ Setup and configure ONNX Runtime in C++
✔️ Use Ort::Env, Ort::SessionOptions, Ort::Session, Ort::Value
✔️ Read model metadata & run inference (Session::Run)
✔️ Optimize models with ORT graph optimizations
✔️ Integrate OpenCV for preprocessing/postprocessing
✔️ Run YOLOv10n object detection in C++
✔️ Query CUDA GPU memory usage


### 🛠️ Troubleshooting

🔹 libonnxruntime.so: cannot open shared object file
➡️ Add ONNX Runtime lib path to runtime linker:

export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH


🔹 opencv2/opencv.hpp: No such file or directory
➡️ Install OpenCV development libraries:

sudo apt-get install libopencv-dev pkg-config


🔹 undefined reference to cuda...
➡️ Ensure CUDA toolkit is installed and visible to compiler:

nvcc --version


If not found, install CUDA from NVIDIA.

🔹 Bounding boxes misaligned in YOLO
➡️ Ensure letterbox preprocessing is applied, and rescale predictions back to original image coordinates.

### 🤝 Contributing

- Fork this repo

- Add your own ONNX Runtime C++ examples

- Submit a PR 🚀

📜 License

MIT License © 2025 Sudhir Silwal