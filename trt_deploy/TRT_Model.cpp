#include "TRT_Model.h"

// ====================1. 构造函数：加载引擎与显存挖坑 ==============
TRTModel::TRTModel(const std::string& engine_path){
    std::cout << " [TRT] 准备读取序列化引擎：" << engine_path << std::endl;

    // 1. 将 .engine 文件以二进制形式读入系统内存（RAM）
    std::ifstream file(engine_path,std::ios::binary);
    if(!file.good()){
        std::cerr << " 找不到引擎文件" << std::endl;
        return;
    }
    file.seekg(0,file.end);
    size_t size = file.tellg();
    file.seekg(0,file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(),size);
    file.close();

    // 2. 召唤三剑客：反序列化生成引擎与上下文
    runtime = nvinfer1::createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engine_data.data(),size);
    context = engine->createExecutionContext();

    // 3. 核心动作：在GPU显存上“挖坑”（分配显存）
    cudaMalloc(&device_buffers[0], INPUT_BYTES);  //挖一个放输入图片的坑
    cudaMalloc(&device_buffers[1],OUTPUT_BYTES);  //挖一个接输出分数的坑

    // 4. 修建一天CUDA异步流（专用的数据传输与计算轨道）
    cudaStreamCreate(&stream);

    std::cout << " [TRT] 引擎加载成功,GPU显存分配完毕!" << std::endl;
}


//  ===================2. 析构函数：大厂规范之手动打扫战场 =============
TRTModel::~TRTModel(){
    //释放GPU显存
    cudaFree(device_buffers[0]);
    cudaFree(device_buffers[1]);
    //销毁流与TRT对象
    cudaStreamDestroy(stream);
    delete context;
    delete engine;
    delete runtime;
    std::cout<< " [TRT]GPU 资源已安全释放" << std::endl;
}

// ================= 3. 批量预处理 (与之前逻辑完全一致，将多图拍扁为一维) =================
void TRTModel::PreprocessBatch(const std::vector<cv::Mat>& images, std::vector<float>& input_tensor_values) {
    int target_height = 224, target_width = 224, channels = 3;
    int image_area = target_height * target_width;
    input_tensor_values.resize(BATCH_SIZE * channels * image_area); // 固定开辟 4 张图的空间
    float mean[] = {0.5f, 0.5f, 0.5f};
    float std[] = {0.5f, 0.5f, 0.5f};

    for (size_t b = 0; b < images.size(); ++b) {
        cv::Mat resized_img, rgb_img, float_img;
        cv::resize(images[b], resized_img, cv::Size(target_width, target_height));
        cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);
        rgb_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        float* batch_ptr = input_tensor_values.data() + b * channels * image_area;
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < target_height; ++h) {
                for (int w = 0; w < target_width; ++w) {
                    batch_ptr[c * image_area + h * target_width + w] =
                        (float_img.at<cv::Vec3f>(h, w)[c] - mean[c]) / std[c];
                }
            }
        }
    }
}

// ================= 4. ⚡ 核心推理：异步显存拷贝与 TRT 推理 =================
std::vector<float> TRTModel::PredictBatch(const std::vector<cv::Mat>& images) {
    // 1. CPU 端准备数据
    std::vector<float> input_tensor_values;
    PreprocessBatch(images, input_tensor_values);

    // 2. ⚡ HostToDevice (H2D)：把 CPU 内存里的图片，全速推入 GPU 的坑里！
    // ⚠️ 注意：这里用的是 Async (异步)，配合 CUDA Stream，主线程发完指令立刻返回，绝不阻塞等待！
    cudaMemcpyAsync(device_buffers[0], input_tensor_values.data(), INPUT_BYTES, cudaMemcpyHostToDevice, stream);

    // 3. ⚡ 发号施令：告诉 TRT 我们刚才挖的坑叫什么名字，并触发推理！
    // TensorRT 10 最新的 V3 接口，显式绑定输入输出
    context->setTensorAddress("input", device_buffers[0]);
    context->setTensorAddress("output", device_buffers[1]);
    context->enqueueV3(stream);

    // 4. ⚡ DeviceToHost (D2H)：把 GPU 计算好的分数，拉回 CPU 内存里
    std::vector<float> output_values(BATCH_SIZE);
    cudaMemcpyAsync(output_values.data(), device_buffers[1], OUTPUT_BYTES, cudaMemcpyDeviceToHost, stream);

    // 5. 🛑 同步！等待这条 CUDA 流上的所有操作 (H2D -> 计算 -> D2H) 全部严格执行完毕
    cudaStreamSynchronize(stream);

    // 6. 返回真实的分数
    std::vector<float> scores(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        scores[i] = output_values[i];
    }
    return scores;
}