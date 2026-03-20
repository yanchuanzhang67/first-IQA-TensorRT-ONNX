#ifndef TRT_MODEL_H
#define TRT_MODEL_H

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

// 1. 继承并实现TRT专属的日志器
class TRTLogger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        // 只打印警告和报错信息，屏蔽其他底层INFO
        if (severity <= Severity::kWARNING){
            std::cout << "[TRT 底层]" << msg << std::endl;
        }
    }
};

class TRTModel{
public:
    TRTModel(const std::string& engine_path);
    ~TRTModel();  // C++核心素养：必须手动清理内存

    void PreprocessBatch(const std::vector<cv::Mat>& images, std::vector<float>& input_tensor_values);
    std::vector<float> PredictBatch(const std::vector<cv::Mat>& images);

    // 2. TensorRT 核心三剑客
    TRTLogger logger;
    nvinfer1::IRuntime* runtime = nullptr;       //运行时（负责把文件变成引擎）
    nvinfer1::ICudaEngine* engine = nullptr;     //引擎本体（厂房）
    nvinfer1::IExecutionContext* context = nullptr;  // 执行上下文（车间主任，负责干活）

    //3. 手动接管GPU显存与异步流
    void* device_buffers[2];   //指针数组：保存GPU上输入和输出的显存地址
    cudaStream_t stream;       // CUDA 异步流：GPU的专属高铁轨道

    //常量：静态Batch = 4 的固定内存大小
    const int BATCH_SIZE = 4;
    const size_t INPUT_BYTES = 4 * 3 * 224 * 224 * sizeof(float);
    const size_t OUTPUT_BYTES = 4 * 1 * sizeof(float);
};

#endif

