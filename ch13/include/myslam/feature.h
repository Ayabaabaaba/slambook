#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {
    // forward declare
    struct Frame; // 帧结构体声明
    struct MapPoint; // 地图点结构体声明

    /**
     * 2D 特征点：三角化后会被关联一个地图点
     */
    struct Feature {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Feature> Ptr; // 定义了类型别名Ptr，后面 Frame::Ptr 即指代 std::shared_ptr<Feature>

        // 由于 Frame 和 MapPoint 的实际shared_ptr持有权归地图 Map。为避免 shared_ptr 的循环引用，使用 weak_ptr
        std::weak_ptr<Frame> frame_; // 持有该feature的frame
        cv::KeyPoint position_; // 2D提取位置
        std::weak_ptr<MapPoint> map_point_; // 关联地图点

        bool is_outlier_ = false; // 是否为异常点
        bool is_on_left_image_ = true; // 标识是否提在左图。false为右图。

        public:
        Feature() {}

        Feature( std::shared_ptr<Frame> frame, const cv::KeyPoint &kp )
            : frame_(frame), position_(kp) {}
    };
}  // namespace myslam

#endif  // MYSLAM_FEATURE_H