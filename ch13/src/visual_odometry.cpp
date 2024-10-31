#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"

namespace myslam {
    VisualOdometry::VisualOdometry(std::string &config_path) : config_file_path_(config_path) {} // 构造函数。接受一个配置文件路径

    bool VisualOdometry::Init() {
        // read from config file
        if (Config::SetParameterFile(config_file_path_) == false) {
            return false;
        }
        
        dataset_ = Dataset::Ptr(new Dataset( Config::Get<std::string>("dataset_dir") )   ); // <std::string>为模板参数，替代T来指定返回类型。
        CHECK_EQ(dataset_->Init(), true); // 检查Dataset对象是否成功初始化。 // CHECK_EQ 用于在调试时检查两个值是否相等，如果它们不相等，则打印一条错误信息并终止程序。

        // create components and links. 创建前端、后端、地图和查看器组件的实例 
        frontend_ = Frontend::Ptr(new Frontend);
        backend_ = Backend::Ptr(new Backend);
        map_ = Map::Ptr(new Map);
        viewer_ = Viewer::Ptr(new Viewer);
        // 设置组件之间的关联
        frontend_->SetBackend(backend_); // 前端设置后端
        frontend_->SetMap(map_);  // 前端设置地图
        frontend_->SetViewer(viewer_); // 前端设置查看器 
        frontend_->SetCameras( dataset_->GetCamera(0), dataset_->GetCamera(1) ); // 前端设置左右目相机。本例只用到了灰度双目数据，对应calib.txt里的头两个相机。
        backend_->SetMap(map_); // 后端设置地图  
        backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1) );  // 后端设置相机  
        viewer_->SetMap(map_); // 查看器设置地图  

        return true;
    }

    void VisualOdometry::Run() { // VO运行
        while(1) { // 无限循环，直到Step()返回false
            LOG(INFO) << "VO is running ";
            if (Step() == false ) { // Step()调用前端处理新一帧数据
                break;
            }
        }

        backend_->Stop(); // 停止后端处理
        viewer_->Close(); // 关闭查看器 

        LOG(INFO) << "VO exit";
    }

    bool VisualOdometry::Step() { // VisualOdometry类的单步处理方法
        Frame::Ptr new_frame = dataset_->NextFrame(); // 从数据集中获取下一帧 
        if (new_frame == nullptr) {
            std::cout << "没有更多帧了" << std::abort;
            return false; // 如果没有更多帧，则返回false
        }
        auto t1 = std::chrono::steady_clock::now();
        bool success = frontend_->AddFrame(new_frame); // 调用前端处理新帧
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1); // 计算处理时间 
        LOG(INFO) << "VO cast time: " << time_used.count() << " seconds.";
        return success;  // 返回前端处理是否成功
    }
} // namespace myslam