#pragma once   //防止头文件被重复引用(面试考点)
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory> // 需要这个来使用 unique_ptr

class IQA_Modal {
public: 
    // 1.构造函数，负责初始化（比如加载模型）
    IQAModel(const  std::string&model_path);

    // 2. 预测接口：输入图片，输出分数
    float Predict(cv::Mat& img);

private:
    // 3. 内部工具：预处理（客人不需要看，所以是private）
    std::vector<float> Preprocess(cv::Mat& img);

    //存放 ONNX Runtime 的环境和会话
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> Session;
};
