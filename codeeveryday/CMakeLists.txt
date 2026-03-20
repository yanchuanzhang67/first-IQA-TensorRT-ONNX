cmake_minimum_required(VERSION 3.10)
project(OnnxInference)

#1. 设置C++标准（c++14或17）
set(CMAKE_CXX_STANDARD 17)

# 2. 告诉 CMake 我们的 ONNX Runtime 在哪里
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/lib/onnxruntime")

# 3. 包含头文件（相当于python 的import search path)
include_directories(${ORT_ROOT}/include)

# 4. 链接库文件目录
link_directories(${ORT_ROOT}/lib)

# 5. 定义可执行文件
add_executable(demo_main day17_inference.cpp)

# 6.链接动态库
# 这里的onnxruntime 就是libonnxruntime.so
target_link_libraries(demo_main onnxruntime)