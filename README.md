# 🚀 ONNX-Tutorial-CPP

**ONNX-Tutorial-C++** is a hands-on guide for learning how to integrate and use **ONNX Runtime** in modern **C++**.  
This repository is designed to run object detection inference with ONNX Runtime in C++ using practical examples.  

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
- `assets/images/` → Place test images here (e.g., `car.jpg`).  
- `assets/output/` → output images here (e.g., `Yolov11_output_car.jpg`). 

---

## 🛠️ Setup Instructions  

**Clone the Repository**
```bash
git clone https://github.com/sudhirsilwal23/ONNX-Tutorial-CPP.git
```

### 1️⃣ Install ONNX Runtime C++  


First, download the pre-built ONNX Runtime binaries. This guide uses the **GPU version for Linux**.

```bash
wget [https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz)
tar -xvzf onnxruntime-linux-x64-gpu-1.18.1.tgz
```
Now, set the environment variable to make it easy to reference the ONNX Runtime files during compilation.

```
export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-gpu-1.18.1
```


### 2️⃣ How to Compile and Run the code

**1. Ort::MemoryInfo Demo**


Compile:
```
g++ -std=c++17 1.Ort_MemoryInfo.cpp \
    -I $ONNXRUNTIME_ROOT/include \
    -L $ONNXRUNTIME_ROOT/lib -lonnxruntime \
    -Wl,-rpath,$ONNXRUNTIME_ROOT/lib \
    -o ort_memory_info
```

Run:
```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ./ort_memory_info
```
**Note:** For all the core ONNX Runtime concept demos (`1.Ort_MemoryInfo.cpp` → `9.Ort_ModelOptimization.cpp`),  
the compilation and execution steps are the same.  

Just replace the filename in the compile command with the file you want to run.

**2. YOLOv10n Object Detection**


Compile:
```
g++ -std=c++17 10.Ort_Detect_YOLOv10n.cpp \
    -I $ONNXRUNTIME_ROOT/include \
    -L $ONNXRUNTIME_ROOT/lib -lonnxruntime \
    `pkg-config --cflags --libs opencv4` \
    -Wl,-rpath,$ONNXRUNTIME_ROOT/lib \
    -o ort_yolo10n
```

Run:
```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ./ort_yolo10n
```

**3. CUDA Memory Info**


Compile:
```
g++ -std=c++17 11.cuda_memory_info.cpp \
    -I $ONNXRUNTIME_ROOT/include \
    -L $ONNXRUNTIME_ROOT/lib -lonnxruntime \
    -lcuda -lcudart \
    -Wl,-rpath,$ONNXRUNTIME_ROOT/lib \
    -o cuda_memory_info
```
Run:
```
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ./cuda_memory_info   
```
---

### 🛠️ Troubleshooting

**🔹 libonnxruntime.so: cannot open shared object file**

➡️ Add ONNX Runtime lib path to runtime linker:

```
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

**🔹 opencv2/opencv.hpp: No such file or directory**

➡️ Install OpenCV development libraries:

```
sudo apt-get install libopencv-dev pkg-config
```

**🔹 undefined reference to cuda...**

➡️ Ensure CUDA toolkit is installed and visible to compiler:

```
nvcc --version
```

**If not found, install CUDA from NVIDIA.**

---

### 🤝 Contributing

- Fork this repo
- Add your own ONNX Runtime C++ examples
- Submit a PR 🚀
- If you find any bug in the code, please report to sudhirsilwal23@gmail.com

---

### 📜 License

MIT License © 2025 Sudhir Silwal

---

### Acknowledgements

- [ONNX Runtime C++ API Documentation](https://onnxruntime.ai/docs/api/c/)  
- [Ultralytics ONNX Integration Guide](https://docs.ultralytics.com/integrations/onnx/)  
- [OpenCV C++ Tutorials](https://www.opencv-srf.com/2017/11/opencv-cpp-api.html)  
