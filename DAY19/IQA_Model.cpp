#include "IQA_Model.h"
#include <iostream>

// --- 1. 实现构造函数 (加载模型) ---
IQAModel::IQAModel(const std::string& model_path) 
    : env(ORT_LOGGING_LEVEL_WARNING, "SDK"), session_options() 
{
    // 加载 ONNX 模型
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
}

// --- 2. 实现预处理 (昨天的核心逻辑) ---
std::vector<float> IQAModel::Preprocess(cv::Mat& img) {
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    std::vector<float> input_values;
    input_values.resize(1 * 3 * 224 * 224);

    float mean[] = {0.485f, 0.456f, 0.406f};
    float std[]  = {0.229f, 0.224f, 0.225f};

    for (int h = 0; h < 224; h++) {
        for (int w = 0; w < 224; w++) {
            cv::Vec3b pixel = resized_img.at<cv::Vec3b>(h, w);
            for (int c = 0; c < 3; c++) {
                float normalized_pixel = (pixel[c] / 255.0f - mean[c]) / std[c];
                int index = c * 224 * 224 + h * 224 + w;
                input_values[index] = normalized_pixel;
            }
        }
    }
    return input_values;
}

// --- 3. 实现预测接口 ---
float IQAModel::Predict(cv::Mat& img) {
    // A. 预处理
    std::vector<float> input_values = Preprocess(img);

    // B. 包装成 Tensor
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_values.size(), 
        input_shape.data(), input_shape.size()
    );

    // C. 推理
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // D. 返回结果
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    return floatarr[0];
}