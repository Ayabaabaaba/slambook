#pragma once

#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam {
    class Map {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType; 
        // std::unordered_map 是一种基于哈希表的关联容器，它允许通过键（key）快速访问值（value），提供了平均常数时间复杂度的查找操作。此处是为了存储地图点和关键帧。

        Map() {}

        void InsertKeyFrame(Frame::Ptr frame);  // 增加一个关键帧
        void InsertMapPoint(MapPoint::Ptr map_point);  // 增加一个地图顶点

        LandmarksType GetAllMapPoints() { // 获取所有地图点
            std::unique_lock<std::mutex> lck(data_mutex_);
            return landmarks_;          }

        KeyframesType GetAllKeyFrames() { // 获取所有关键帧
            std::unique_lock<std::mutex> lck(data_mutex_);
            return keyframes_;          }

        LandmarksType GetActiveMapPoints() { // 获取激活地图点
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_landmarks_;      }

        KeyframesType GetActiveKeyFrames() {  // 获取激活关键帧
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_keyframes_;      }

        void CleanMap(); // 清理 map 中观测数量为零的点

        private:
        void RemoveOldKeyframe(); // 将旧的关键帧置为不活跃状态
        std::mutex data_mutex_;
        LandmarksType landmarks_;  // all landmarks
        LandmarksType active_landmarks_; // active landmarks
        KeyframesType keyframes_;  // all key-frames
        KeyframesType active_keyframes_; // active key-frames

        Frame::Ptr current_frame_ = nullptr; // current_frame_ 成员变量被初始化为 nullptr，表示在 Map 对象被创建时没有与之关联的当前帧。

        int num_active_keyframes_ = 7; // 激活的关键帧数量（自定义）

    }; // class Map
}  // namespace myslam

#endif  // MAP_H