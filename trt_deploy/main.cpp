#include "TRT_Model.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <deque>

// ================= 全局变量：传送带与调度锁 =================
std::queue<cv::Mat> frame_queue;
std::mutex mtx;
std::condition_variable cv_queue;
std::atomic<bool> is_reading_finished(false);

// ================= 🔪 配菜员：生产者副线程 =================
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
        // 控制传送带长度，防止把系统内存撑爆
        cv_queue.wait(lock, []{ return frame_queue.size() < 30; });
        frame_queue.push(frame.clone()); 
        lock.unlock();
        cv_queue.notify_one();
    }

    is_reading_finished = true;
    cv_queue.notify_all();
    cap.release();
}

// ================= 👨‍🍳 主厨：消费者主线程 =================
int main() {
    std::cout << "🚀 [系统] 正在初始化 TRT 纯血异步高并发引擎..." << std::endl;
    
    // 🌟 核心替换：加载我们刚刚用 Python 锻造好的极限 FP16 引擎！
    TRTModel model("maniqa_batch4_fp16.engine");

    std::string video_path = "test_maidou.mp4";
    cv::VideoCapture temp_cap(video_path);
    if (!temp_cap.isOpened()) return -1;
    int frame_width = temp_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = temp_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double original_fps = temp_cap.get(cv::CAP_PROP_FPS);
    int total_frames = temp_cap.get(cv::CAP_PROP_FRAME_COUNT);
    temp_cap.release();

    cv::VideoWriter writer("output_trt_result.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), original_fps, cv::Size(frame_width, frame_height));

    // 启动异步读图线程
    std::thread reader_thread(VideoReaderThread, video_path);

    int frame_count = 0;
    double total_latency_ms = 0.0;
    std::deque<float> score_buffer;
    const int SMOOTH_WINDOW = 10;
    const int BATCH_SIZE = 4; // 严格对齐静态模型

    std::cout << "⏳ [系统] TRT V8 引擎就绪，流水线全速运转中..." << std::endl;

    while (true) {
        std::vector<cv::Mat> batch_frames;

        std::unique_lock<std::mutex> lock(mtx);
        cv_queue.wait(lock, [&]{ 
            return frame_queue.size() >= BATCH_SIZE || (is_reading_finished && !frame_queue.empty()) || (frame_queue.empty() && is_reading_finished); 
        });

        if (frame_queue.empty() && is_reading_finished) break; 

        while (!frame_queue.empty() && batch_frames.size() < BATCH_SIZE) {
            batch_frames.push_back(frame_queue.front());
            frame_queue.pop();
        }
        lock.unlock();
        cv_queue.notify_all(); 

        // ================= 🚀 TRT 静态 Batch 压榨与尾帧补齐 =================
        auto start_time = std::chrono::high_resolution_clock::now();

        int valid_frames = batch_frames.size(); 
        while (batch_frames.size() < BATCH_SIZE) {
            batch_frames.push_back(batch_frames.back().clone()); 
        }

        // 🧠 终极调用：一行代码触发 CUDA 异步流、显存拷贝与 TRT 推理！
        std::vector<float> raw_scores = model.PredictBatch(batch_frames);

        raw_scores.resize(valid_frames);
        batch_frames.resize(valid_frames);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> latency = end_time - start_time;
        // =========================================================

        double batch_latency = latency.count();
        double current_fps = (1000.0 / batch_latency) * valid_frames; 
        total_latency_ms += batch_latency;
        frame_count += valid_frames; 

        for (size_t i = 0; i < batch_frames.size(); ++i) {
            score_buffer.push_back(raw_scores[i]);
            if (score_buffer.size() > SMOOTH_WINDOW) score_buffer.pop_front();
            float sum = 0;
            for (float s : score_buffer) sum += s;
            float smooth_score = sum / score_buffer.size();

            std::string text_score = "TRT IQA: " + std::to_string(smooth_score);
            std::string text_fps = "TRT FPS: " + std::to_string(current_fps);
            cv::putText(batch_frames[i], text_score, cv::Point(30, 250), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
            cv::putText(batch_frames[i], text_fps, cv::Point(30, 300), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

            writer.write(batch_frames[i]);
        }

        std::cout << "进度: " << frame_count << "/" << total_frames 
                  << " | GPU批耗时: " << batch_latency << " ms" 
                  << " | TRT峰值FPS: " << current_fps << std::endl;
    }

    reader_thread.join();
    writer.release();

    double avg_latency = total_latency_ms / frame_count;
    double avg_fps = 1000.0 / avg_latency;

    std::cout << "\n=============================================" << std::endl;
    std::cout << "🏆 [TRT 极限报告] 纯 GPU 推理平均耗时: " << avg_latency << " ms" << std::endl;
    std::cout << "🏆 [TRT 极限报告] 异步架构系统峰值 FPS: " << avg_fps << std::endl;
    std::cout << "=============================================\n" << std::endl;

    return 0;
}