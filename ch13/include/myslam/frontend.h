#pragma once

#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
    class Backend; // 声明Backend和Viewer类。（前向声明：用于在Frontend类中引用这些类，而无需包含它们的完整定义）
    class Viewer;

    enum class FrontendStatus { INITING, TRACKING_GOOD, TRACKING_BAD, LOST }; // 定义一个枚举类FrontendStatus，用于表示前端的状态，包括初始化中、跟踪良好、跟踪不佳和丢失。
    // 枚举类enum：一种特殊的数据类型，用{}包含一组预定义的常量值。
    // enum 传统枚举：作用域通常是全局的，除非被显式地封装在某作用域内。可以隐式地转换为整数类型。
    // enum class 作用域限定的枚举：具有类似于类的作用域，其成员仅在枚举类内部可见。无隐藏式转换。（由于作用域限制和无隐式转换的特性，具有更强的类型安全保证）

    /**
     * 前端：估计当前帧 Pose，在满足关键帧条件时向地图加入关键帧并触发优化
     */
    class Frontend {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 确保使用Eigen类型的对象时，动态内存分配（如new操作）是对齐的
        typedef std::shared_ptr<Frontend> Ptr; // 定义一个类型别名Ptr：Frontend类的std::shared_ptr智能指针。

        Frontend(); // 构造函数
 
        bool AddFrame( Frame::Ptr frame ); // 添加一个帧并计算其定位结果

        void SetMap(Map::Ptr map) { map_ = map; } // Set 函数，用于设置地图

        void SetBackend( std::shared_ptr<Backend> backend ) {backend_ = backend; } // 设置后端。

        void SetViewer( std::shared_ptr<Viewer> viewer ) { viewer_ = viewer; } // 设置 viewer 视图器。

        FrontendStatus GetStatus() const { return status_; } // 返回前端的状态。const关键字表示这个函数不会修改类的任何成员变量。
        // const 在末尾：表示常量成员函数（只读成员函数），不会修改函数内部的非静态成员变量（除非被声明为 mutable）。通常用于读取对象信息。
        // const 在开头：表示返回值是一个常量值。但不影响函数对成员变量的可修改性。

        void SetCameras( Camera::Ptr left, Camera::Ptr right ) { // 用于设置左右相机
            camera_left_ = left;
            camera_right_ = right;
        }

        private: 
        // 跟踪、重置、估计当前位姿等。
        bool Track(); // Track in normal mode. @return true if success
        bool Reset(); // Reset when lost. @return true if success
        int TrackLastFrame(); // Track with last frame. @return num of tracked points.
        int EstimateCurrentPose(); // estimate current frame's pose. @return num of inliers.
        bool InsertKeyframe(); // set current frame as a keyframe and insert it into backend.
        bool StereoInit(); // Try init the frontend with stereo images saved in current_frame_. @return true if success.
        int DetectFeatures(); // Detect features in left image in current_frame_. Keypoints will be saved in current_frame_.
        int FindFeaturesInRight(); // Find the corresponding features in right image of current_frame_. @return num of features found.
        bool BuildInitMap(); // Build the initial map with single image. @return true if succeed.
        int TriangulateNewPoints(); // Triangulate the 2D points in current frame. @return num of triangulated points.
        void SetObservationsForKeyFrame(); // Set the features in keyframes as new observation of the map points

        FrontendStatus status_ = FrontendStatus::INITING; // Data. 用于存储前端的状态

        Frame::Ptr current_frame_ = nullptr; // 当前帧
        Frame::Ptr last_frame_ = nullptr; // 上一帧
        Camera::Ptr camera_left_ = nullptr; // 左侧相机
        Camera::Ptr camera_right_ = nullptr; // 右侧相机

        Map::Ptr map_ = nullptr; // 地图
        std::shared_ptr<Backend> backend_ = nullptr; // 后端
        std::shared_ptr<Viewer> viewer_ = nullptr; // 视图器

        SE3 relative_motion_; // 当前帧与上一帧的相对运动，用于估计当前帧 pose 初值。似乎这样定义后，默认为单位阵？

        int tracking_inliers_ = 0; // inliers, used for testing new keyframes. 跟踪窗

        // params. 特征点数量等
        int num_features_ = 200;
        int num_features_init_ = 100; // 正确初始化时，检测出的特征点数量的下限
        int num_features_tracking_ = 50; // 跟踪良好状态的特征点数量边界
        int num_features_tracking_bad_ = 20; // 跟踪不良状态的特征点数量边界。小于这个值则视为丢失。
        int num_features_needed_for_keyframe_ = 80;

        // utilities
        cv::Ptr<cv::GFTTDetector> gftt_; // feature detector in opencv. 定义一个OpenCV中的特征检测器gftt_，用于检测图像中的特征点。
    
    }; // class Frontend
} // namespace myslam

#endif // MYSLAM_FRONTEND_H