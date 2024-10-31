#ifndef MYSLAM_VIEWER_H
#define MYSLAM_VIEWER_H

#include <thread>
#include <pangolin/pangolin.h> // 实时3D可视化工具
#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
    /**
     * 可视化
     */
    class Viewer {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Viewer> Ptr; // 定义一个类型别名Ptr，为Viewer类的std::shared_ptr智能指针类型。智能指针有助于管理动态分配的内存，自动释放不再需要的内存
        
        Viewer(); // 构造函数
        
        void SetMap( Map::Ptr map ) { map_ = map; } // 用于设置与Viewer关联的地图

        void Close(); // 可能用于关闭可视化窗口或停止可视化线程

        void AddCurrentFrame( Frame::Ptr current_frame ); // 增加一个当前帧

        void UpdateMap(); // 更新地图

        private:
        void ThreadLoop(); // 可能是用于可视化更新的线程的主循环

        void DrawFrame( Frame::Ptr frame, const float* color ); // 绘制当前帧

        void DrawMapPoints(); // 绘制地图点

        void FollowCurrentFrame( pangolin::OpenGlRenderState& vis_camera ); // 使视图跟随当前帧

        // plot the features in current frame into an image
        cv::Mat PlotFrameImage(); // 将当前帧的特征绘制到一张图像上

        Frame::Ptr current_frame_ = nullptr; // 存储当前帧的指针
        Map::Ptr map_ = nullptr; // 存储地图的指针

        std::thread viewer_thread_ ; // 存储可视化线程的线程对象
        bool viewer_running_ = true; // 用于指示可视化线程是否应该继续运行。

        // 用于存储活跃的关键帧和地图点的映射
        std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_; 
        std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;
        bool map_updated_ = false; // 一个标志，用于指示地图是否已更新

        std::mutex viewer_data_mutex_;  // 一个互斥锁，用于保护对Viewer类内部数据的并发访问。
    }; // class Viewer
} // namespace myslam


#endif // MYSLAM_VIEWER_H