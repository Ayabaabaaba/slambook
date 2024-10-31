#pragma once

#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam {
    // forward declare
    struct Frame;  // 帧结构体声明
    struct Feature; // 特征结构体声明

    /**
     * 路标点类：特征点在三角化后形成的路标点
     */
    struct MapPoint {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        typedef std::shared_ptr<MapPoint> Ptr;  // 定义了类型别名Ptr

        unsigned long id_ = 0; // 存储路标点的唯一标识符 ID
        bool is_outlier_ = false; // 标记这个路标点是否是一个外点
        Vec3 pos_ = Vec3::Zero();  // Position in world. 存储路标点在世界坐标系中的位置。
        std::mutex data_mutex_; // 互斥锁，用于在多线程下保护MapPoint成员
        int observed_times_ = 0; // being observed by feature matching algo. 记录这个路标点被特征匹配算法观测到的次数。
        std::list<std::weak_ptr<Feature>> observations_; // 定义一个std::list容器，存储std::weak_ptr<Feature>类型的元素。弱引用可避免循环引用导致的内存泄漏。
        // std::list 不同于 std::vector ，在内存上不必连续，插入或删除操作较快。

        MapPoint() {}
        MapPoint(long id, Vec3 position);

        Vec3 Pos() { // 用于获取路标点位置的成员函数。
        // 在访问pos_之前，先使用std::unique_lock加锁，确保线程安全。
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void SetPos( const Vec3 &pos ) { // 用于设置路标点位置的成员函数
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        }

        void AddObservation(std::shared_ptr<Feature> feature) {
        // 向observations_列表中添加一个新的特征观测，并增加观测次数
            std::unique_lock<std::mutex> lck(data_mutex_);
            observations_.push_back(feature);
            observed_times_++;
        }

        void RemoveObservation( std::shared_ptr<Feature> feat); // 用于从observations_列表中移除一个特征观测。定义在.cpp

        std::list<std::weak_ptr<Feature>> GetObs() {
        // 用于获取所有观测到这个路标点的特征点的弱引用列表
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        // factory function
        static MapPoint::Ptr CreateNewMappoint(); // 用于创建一个新的MapPoint对象，并返回其智能指针。
        // 通常这样的工厂函数用于统一创建对象，可以在创建过程中进行一些初始化操作。
    };
} // namespace myslam

#endif // MYSLAM_MAPPOINT_H