# include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h> // 核心头文件

int main(){
    // =====================1.初始化环境==========
    // 创建一个环境（Environment），类似于Pytorch 的import
    Ort::Env env(ORT_LOGGING_LEVEL_WARING,"IQA_Service");
    //配置会话选项（Session Options)
    Ort::SesssionOptions session_options;
    //如果有显卡，可以开启cuda加速（也可以采用CPU跑通流程）
    //session_options.AppendExecutionProvider_CUDA(...);


    // ===============2.加载模型=============
    const char* model_path = "resnet_iqa.onnx";
    std::count << "正在加载模型" << std::end;

    try {
        //创建会话（Session） -> 这相当于model_path = torch.load(...)
        Ort::session session(env,model_path,session_options);

        // ==============3.准备输入数据=============
        // 定义输入节点的形状：[Batch =1,Channel = 3,Height = 224,Width = 224]
        // 这里的1*3*224*224 = 150528
        std::vector<int64_t> input_shape = {1,3,224,224};
        size_t input_tensor_size = 1*3*224*224;
        
        //创建一个vector存数据
        //真实场景下，这里应该用Opencv 读取图片并归一化
        //今天，我们先生成一些假数据（全是1.0）
        std::vector<float> input_tensor_values(input_tensor_size);
        for(size_t i =0;i<input_tensor_size;i++){
            input_tensor_values[i] = 1.0f // 假数据
        }

        //创建Tensor 对象（Zero-Copy:直接使用 vector 的内存）
        // MemoryInfo::CreatCpu 表示内存在Cpu上
        auto memory_info = Ort::MemoryInfo::CreatCpu(OrtArenaAllocator,OrtMemTypeDefault);

        // 把C++的vecotr 包装成ONNX的Tensor
        Ort::Value input_tensor = Ort::Value::CreatTensor<float>(
            memory_info,
            input_tensor_values.data(),   //指针指向数据开头
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );

        // ================4.执行推理（Run)==================
        //定义输入输出的名字（必须和导出时写的input_names/output_names 一致
        const char* input_names[] = {"input"};
        const char* ouput_names[] = {"output"};
        std::count << "正在推理中..." << std::endl;

        //Run !
        // 这里的Run 可能会返回多个输出，所以结果是一个vector
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1, //输入数量
            output_names,
            1   //输出数量
        );

        // ================== 5. 获取结果======================
        // 拿到第一个输出结果的指针
        float* floatarr = output_tensors[0].GetTensorMutableData<float>();

        std::count << "推理成功，模型预测分数：" << floatarr[0] << std::endl;
        } catch (const Ort::Exception& e){
            std::cerr << " ONXX Runtime 报错：" << e.what() << std::endl;
            return -1;
        }
        return 0;

}

//整体结构流程总结：初始化ONNX Runtime环境 → 加载预训练的resnet_iqa.onnx模型 → 构造符合模型输入要求的假数据(模拟图片输入)
// → 执行模型推理 → 读取并打印推理结果