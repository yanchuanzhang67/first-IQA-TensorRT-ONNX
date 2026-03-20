cmake_minimum_required(VERSION 3.10)
# 项目名称：TRT极限异步流水线
project(TRT_IQA_Pipeline)

# 强制使用 C++17 标准 (支持多线程等高级特性)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ================= 1. 引入 OpenCV =================
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ================= 2. 引入 CUDA 12.2 =================
# 指定 CUDA 的头文件和库文件路径
set(CUDA_DIR "/usr/local/cuda-12.2")
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

# ================= 3. 引入 TensorRT 10 =================
# 指定我们刚刚下载解压的 TRT 军火库路径
set(TRT_DIR "/workspace/TensorRT")
include_directories(${TRT_DIR}/include)
link_directories(${TRT_DIR}/lib)

# ================= 4. 编译可执行文件 =================
# 把 main.cpp 和 TRT_Model.cpp 编译成名为 trt_app 的程序
add_executable(trt_app main.cpp TRT_Model.cpp)

# ================= 5. 终极链接 =================
# 将 OpenCV、TensorRT核心(nvinfer) 和 CUDA运行时(cudart) 链接到程序中
target_link_libraries(trt_app
    ${OpenCV_LIBS}
    nvinfer
    cudart
)