#include "IQA_Model.h" // 只导入菜单
#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
    // 1. 实例化：这行代码一跑，模型就加载好了
    std::cout << "正在初始化 IQA模型..." << std::end;
    IQA_Model model("resnet_iqa.onnx")
    
    //2. 准备图片 (假装从磁盘中读了一张图)
    cv::Mat img(500,500,CV_8UC3,cv::Scalar(0,0,225));

    //3. 只需要调这一个函数，不用管内部多复杂
    float score = model.Predict(img);

    std::cout << "最终得分：" << score << std::endl;
    return 0;
}