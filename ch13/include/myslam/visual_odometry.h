#pragma once

#ifndef MYSLAM_VISUAL_ODOMETRY_H
#define MYSLAM_VISUAL_ODOMETRY_H

#include "myslam/backend.h"
#include "myslam/common_include.h"
#include "myslam/dataset.h"
#include "myslam/frontend.h"
#include "myslam/viewer.h"

namespace myslam {
    /**
     * VO 对外接口
     */
    class VisualOdometry {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<VisualOdometry> Ptr;

        VisualOdometry(std::string &config_path); // constructor with config file

        bool Init(); // do initialization things before run. @return true if success.

        void Run(); // start vo in the dataset.

        bool Step(); // Make a step forward in dataset.

        FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); } // 获取前端状态

        private:
        bool inited_ = false; // 用于标记VisualOdometry对象是否已经完成初始化。完成初始化后设置为true，可防止重复初始化。
        
        // VisualOdometry类加载配置、初始化前端和后端、管理地图和查看器，并在数据集上运行视觉里程计算法
        std::string config_file_path_; // 配置文件的路径
        Frontend::Ptr frontend_ = nullptr; // 指向Frontend类的智能指针
        Backend::Ptr backend_ = nullptr;
        Map::Ptr map_ = nullptr;
        Viewer::Ptr viewer_ = nullptr;
        Dataset::Ptr dataset_ = nullptr; // dataset
    }; // class VisualOdometry
} // namespace myslam

#endif // MYSLAM_VISUAL_ODOMETRY_H