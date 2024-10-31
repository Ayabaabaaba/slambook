#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {
    // forward declare
    struct MapPoint;  // 地图点结构体声明
    struct Feature; // 特征结构体声明

    /**
     * 帧：每一帧被分配独立id，关键帧分配关键帧ID
     */
    struct Frame {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr; // 定义类型别名Ptr

        unsigned long id_ = 0; // id of this frame
        unsigned long keyframe_id_ = 0; // id of key frame
        bool is_keyframe_ = false; // 是否为关键帧
        double time_stamp_; // 时间戳，暂不使用
        SE3 pose_; // Tcw形式Pose，世界系->当前帧。似乎这个定义完就默认是单位旋转、零平移了？
        std::mutex pose_mutex_; // Pose数据锁。 用于保护 pose_ 成员变量（存储帧的位姿）。
        // std::mutex 为互斥锁对象，保证多线程访问时，同一时间只有一个线程可访问相应变量。
        cv::Mat left_img_, right_img_; // stereo images

        // extracted features in left image
        std::vector< std::shared_ptr<Feature> > features_left_;
        // corresponding features in right image, set to nullptr if no corresponding
        std::vector< std::shared_ptr<Feature> > features_right_;

        public: // data members
        Frame() {}
        Frame( long id, double time_stamp, const SE3 &pose, const Mat &left, const Mat &right );
        // set and get pose, thread safe
        SE3 Pose() {
            std::unique_lock<std::mutex> lck(pose_mutex_); // 创建一个 std::unique_lock 对象 lck，立即锁定了 pose_mutex_。
            // 在 lck 的作用域内，没有其他线程可以修改 pose_，确保了 pose_ 的访问是线程安全的。
            return pose_;
            // 当函数返回时，lck 对象超出作用域并被销毁，此时会隐式释放 pose_mutex_ 锁  
            // 显式释放可用： lck.unlock(); 需注意，一旦调用unlock()，lck就不再拥有锁，再次调用lock()会报错。
        }
        void SetPose( const SE3 &pose ) { 
            std::unique_lock<std::mutex> lck(pose_mutex_); 
            pose_ = pose;
        }
        // 设置关键帧并分配关键帧ID
        void SetKeyFrame(); 
        // 工厂构建模式，分配ID
        static std::shared_ptr<Frame> CreateFrame(); // 构建函数的返回类型为 std::shared_ptr<Frame> 指向 Frame型对象的智能指针。前面typedef定义了别名Ptr。
        // CreateFrame() 为工厂构建模式，用于创建新的 Frame 类型的实例并分配唯一ID，如果不是静态的则没有意义。
    };

} // namespace myslam

#endif // MYSLAM_FRAME_H