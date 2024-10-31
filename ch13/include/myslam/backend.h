#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
    class Map;

    /**
     * 后端：有单独优化线程，在Map更新时启动优化
     * Map更新由前端触发
     */
    class Backend {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Backend> Ptr;

        Backend(); // 构造函数：启动优化线程并挂起

        void SetCameras(Camera::Ptr left, Camera::Ptr right) { // 设置左右目相机，用于获得内外参
            cam_left_ = left;
            cam_right_ = right;
        }

        void SetMap(std::shared_ptr<Map> map ) { map_ = map; } // 设置地图

        void UpdateMap(); // 触发地图更新，启动优化

        void Stop(); // 关闭后端线程

        private:
        void BackendLoop(); // 后端线程

        void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks); // 对给定关键帧和路标点进行优化

        // 分别用于存储地图对象、后端线程和数据互斥锁。
        std::shared_ptr<Map> map_;
        std::thread backend_thread_;
        std::mutex data_mutex_;

        // 分别用于条件变量（用于线程间的同步）和一个原子布尔值（用于标记后端线程是否正在运行）
        std::condition_variable map_update_; // std::condition_variable 是 C++11 的一个同步原语，用于实现线程间的同步。它允许一个或多个线程在某个条件满足之前等待，而另一个线程可以在条件满足时通知等待的线程继续执行。
        std::atomic<bool> backend_running_; // std::atomic 是 C++11 的模板类，用于实现无需锁（lock-free）的原子操作。原子操作不可被中断，即多线程环境下，一个线程执行原子操作时，其他线程无法看到该操作进行到一半的状态，只能看到操作之前或之后的完整状态。

        Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr; // 分别用于存储左右目相机的智能指针
    }; // class Backend
} // namespace myslam

#endif // MYSLAM_BACKEND_H