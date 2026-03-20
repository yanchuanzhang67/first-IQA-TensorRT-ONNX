#include "IQA_Model.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>             // C++ 多线程支持
#include <mutex>              // 互斥锁，保护传送带
#include <queue>              // 队列，也就是我们的“传送带”
#include <condition_variable> // 条件变量，用于线程间互相呼叫
#include <atomic>             // 原子操作，用于安全标记视频是否读完
#include <deque>              // 双端队列，用于分数平滑处理

// ================= 全局变量：传送带与调度锁 =================
std::queue<cv::Mat> frame_queue;
std::mutex mtx;
std::condition_variable cv_queue;
std::atomic<bool> is_reading_finished(false);

// ================= 🔪 配菜员：生产者线程 =================
void VideoReaderThread(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        is_reading_finished = true;
        cv_queue.notify_all();
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        std::unique_lock<std::mutex> lock(mtx);
        
        // 如果传送带上堆了超过 30 帧，配菜员就停下来歇会儿，防止把服务器内存撑爆
        cv_queue.wait(lock, []{ return frame_queue.size() < 30; });

        // 将画面放上传送带！⚠️ 必须用 clone()，防止 OpenCV 底层复用内存导致画面撕裂
        frame_queue.push(frame.clone()); 
        
        lock.unlock();
        cv_queue.notify_one(); // 呼叫主厨(GPU)：有新画面来了！
    }

    is_reading_finished = true;
    cv_queue.notify_all(); // 告诉主厨：视频读完了，准备下班
    cap.release();
}

// ================= 👨‍🍳 主厨：消费者线程 (主线程) =================
int main() {
    std::cout << "🚀 [系统] 正在初始化 IQA 异步高并发引擎..." << std::endl;
    IQAModel model("resnet_iqa.onnx");

    std::string video_path = "test_maidou.mp4";

    // 先粗略读一下视频属性，给 VideoWriter 准备参数
    cv::VideoCapture temp_cap(video_path);
    if (!temp_cap.isOpened()) return -1;
    int frame_width = temp_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = temp_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double original_fps = temp_cap.get(cv::CAP_PROP_FPS);
    int total_frames = temp_cap.get(cv::CAP_PROP_FRAME_COUNT);
    temp_cap.release();

    cv::VideoWriter writer("output_result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), original_fps, cv::Size(frame_width, frame_height));

    // 🌟 核心魔法：启动生产者副线程！它会在后台默默疯狂读图
    std::thread reader_thread(VideoReaderThread, video_path);

    int frame_count = 0;
    double total_latency_ms = 0.0;

    // 🌟 UI 优化：用于平滑分数的缓存池 (缓存过去 10 帧的分数)
    std::deque<float> score_buffer;
    const int SMOOTH_WINDOW = 10;

    std::cout << "⏳ [系统] GPU 引擎就绪，流水线全速运转中..." << std::endl;

    while (true) {
        cv::Mat frame;
        
        // 主厨盯着传送带看
        std::unique_lock<std::mutex> lock(mtx);
        cv_queue.wait(lock, []{ return !frame_queue.empty() || is_reading_finished; });

        // 如果传送带空了，且配菜员下班了，说明全剧终
        if (frame_queue.empty() && is_reading_finished) {
            break; 
        }

        // 从传送带拿下画面
        frame = frame_queue.front();
        frame_queue.pop();
        lock.unlock();
        cv_queue.notify_all(); // 呼叫配菜员：传送带有空位了，继续切！

        // ================= 开始纯 GPU 性能压榨 =================
        auto start_time = std::chrono::high_resolution_clock::now();

        // 🧠 核心推理
        float raw_score = model.Predict(frame);

        // 📊 平滑滤波：让 UI 上的分数不再疯狂跳动
        score_buffer.push_back(raw_score);
        if (score_buffer.size() > SMOOTH_WINDOW) score_buffer.pop_front();
        float sum = 0;
        for (float s : score_buffer) sum += s;
        float smooth_score = sum / score_buffer.size();

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> latency = end_time - start_time;
        
        double current_latency = latency.count();
        double current_fps = 1000.0 / current_latency; 

        total_latency_ms += current_latency;
        frame_count++;

        // 🎨 OSD 渲染与写入
        std::string text_score = "IQA Score: " + std::to_string(smooth_score);
        std::string text_fps = "GPU FPS: " + std::to_string(current_fps);
        cv::putText(frame, text_score, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, text_fps, cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        writer.write(frame);

        if (frame_count % 10 == 0) {
            std::cout << "进度: " << frame_count << "/" << total_frames 
                      << " | 平滑得分: " << smooth_score 
                      << " | GPU耗时: " << current_latency << " ms" 
                      << " | 峰值FPS: " << current_fps << std::endl;
        }
    }

    // 安全回收副线程（大厂代码规范：必须 join 确保资源释放）
    reader_thread.join();
    writer.release();

    double avg_latency = total_latency_ms / frame_count;
    double avg_fps = 1000.0 / avg_latency;

    std::cout << "\n=============================================" << std::endl;
    std::cout << "📈 [性能报告] 纯 GPU 推理平均耗时: " << avg_latency << " ms" << std::endl;
    std::cout << "📈 [性能报告] 异步架构系统峰值 FPS: " << avg_fps << std::endl;
    std::cout << "=============================================\n" << std::endl;

    return 0;
}