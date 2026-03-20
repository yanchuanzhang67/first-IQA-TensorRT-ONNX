#include <iostream>
#include <vector>
#include <string>
//#include <opencv2/opencv.hpp> // OpenCV 头文件
//#include <onnxruntime_cxx_api.h>

//  ===========核心考点：预处理函数===========
// 目标：将CV::Mat(HWC,BGR,225) -> std::vector(CHW,RGB,Normalized)
std::vector<float> Preprocess(cv::Mat& img) {
    //1.先缩放到模型要求的224x224
    cv::Mat resized_img;
    cv::resize(img,resized_img,cv::Size(224,224));

    //2.对于BGR -> RGB
    cv::cvtColor(resized_img,resized_img,cv::CO:OR_BGR2RGB);

    //3.归一化 + HWC 转CHW
    //准备一个vector 存数据
    std::vector<float> input_tensor_values;
    //预分配内存，防止push_back 导致扩容性能损耗
    input_tensor_values.resize(1*3*224*224);
    
    //Imagenet的均值和方差（训练transforms用的就是这个）
    float mean[] = {0.485,0.456,0.406};
    float std[] = { 0.229,0.224,0.225};

    //遍历像素（手动将HWC 排列成CHW 排列）
    for (int h= 0;h<224;h++){
        for(int w=0;w<224;w++){
            //获取当前像素的RGB值
            cv::Vec3b pixel = resized_img.at<cv::Vec3b>(h,w);

            for(int c= 0;c <3;c++){  //c=0(R) c = 1(G) c= 2(B)
                //A .归一化
                float normalized_pixel = (pixel[c] / 225.0f - mean[c]) /std[c];

                //B .填入vector（CHW格式）
                int index = c*224*224 + h*224 +w;
                input_tensor_values[index] = normalized_pixel;
            }
        }
    }
    return input_tensor_values;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "IQA_Service");
    Ort::SessionOptions session_options;

    //加载模型
    const char* model_path = "resnet_iqa.onnx";
    Ort::Session session(env, model_path, session_options);

    // =======1.读取真实图片=========
    //这里用OpenCV 生成一张纯红色的图来测试
    cv::Mat img = imread("D:\组会汇报\组会汇报\图像基础学习\kunkun.png");

    if(img.empty()) {
        std::cerr << "图片读取失败！"  << std::endl;
        return -1;
    }
    std::cout  << "图片读取成功：" << img.rows << "x" << img.cols << std::endl;

    //  ===========2. 预处理=======
    std::vector<float> input_tensor_values = Preprocess(img);

    // ==========3.创建Tensor=======
    std::vector<int64_t> input_shape = {1,3,224,224};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
    );

    // =========4.推理=========
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "🎉 真实图片推理得分: " << floatarr[0] << std::endl;

    return 0;
}