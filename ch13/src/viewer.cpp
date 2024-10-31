#include "myslam/viewer.h"
#include "myslam/feature.h"
#include "myslam/frame.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam {
    Viewer::Viewer() { // 构造函数
        viewer_thread_ = std::thread( std::bind(&Viewer::ThreadLoop, this) );
        // std::thread 用于创建新线程，该线程将运行ThreadLoop成员函数。
        // std::bind 用于将成员函数与其对象实例绑定。
        // this 指针在类的非静态成员函数中自动存在，指向调用该成员函数的对象实例。
    }

    void Viewer::Close() { // 用于关闭Viewer
        viewer_running_ = false; // 将viewer_running_标志设置为false，以通知ThreadLoop停止运行
        viewer_thread_.join(); // 调用join等待viewer线程结束。
        // .join 函数：告诉调用 Close() 的线程，等待 viewer_thread_ 完成。只有当 ThreadLoop 函数返回，即 viewer_thread_ 线程结束时，.join() 调用才会返回。
    }

    void Viewer::AddCurrentFrame( Frame::Ptr current_frame ) { // 用于添加当前帧
        std::unique_lock<std::mutex> lck(viewer_data_mutex_); // 使用互斥锁viewer_data_mutex_来确保线程安
        current_frame_ = current_frame; // 将传入的帧保存到current_frame_成员变量
    }

    void Viewer::UpdateMap() { // 用于更新地图
        std::unique_lock<std::mutex> lck(viewer_data_mutex_); // 使用互斥锁来确保线程安全
        assert(map_ != nullptr);
        // 从地图对象中获取活跃的关键帧和地图点
        active_keyframes_ = map_->GetActiveKeyFrames();
        active_landmarks_ = map_->GetActiveMapPoints();
        map_updated_ = true; // 标记地图已更新
    }

    void Viewer::ThreadLoop() { // Viewer 的主要工作函数，在新线程中运行
        // 初始化Pangolin
        pangolin::CreateWindowAndBind("MYSLAM", 1024, 768); // 创建一个名为 "MYSLAM" 的窗口，并设置其大小为 1024x768 像素。
        glEnable(GL_DEPTH_TEST); // 启用深度测试，以便正确处理深度信息
        glEnable(GL_BLEND);  // 启用混合（blending），以便正确处理透明效果
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA ); // 设置混合函数

        pangolin::OpenGlRenderState vis_camera( // 设置相机的投影矩阵和模型视图矩阵。
            pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000), // 设置相机的位置和观察方向
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0)    );
        
        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& vis_display = pangolin::CreateDisplay() // 创建一个显示区域，并设置其边界和3D处理器（相机参数 vis_camera）。
                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                .SetHandler(  new pangolin::Handler3D(vis_camera)  );
        
        // 定义蓝色和绿色的RGB值。
        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

        while (!pangolin::ShouldQuit() && viewer_running_) { // 一个标准的 OpenGL 渲染循环。直到用户请求退出（例如，关闭窗口），或运行标志位为 false
        // pangolin::ShouldQuit() 当用户请求退出（如按Esc、关闭窗口等）时，返回true
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );  // 清除屏幕和深度缓冲区
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 设置OpenGL的清除颜色为纯白（即屏幕颜色）
            vis_display.Activate(vis_camera); // 激活显示区域和相机

            std::unique_lock<std::mutex> lock(viewer_data_mutex_); // 使用互斥锁确保线程安全。

            if (current_frame_) {
                // 如果当前帧存在，则绘制当前帧并让相机跟随当前帧。
                DrawFrame( current_frame_, green );
                FollowCurrentFrame( vis_camera );

                cv::Mat img = PlotFrameImage(); // 绘制当前帧的图像
                cv::imshow("imgae", img); // 使用OpenCV显示
                cv::waitKey(1); 
            }

            if (map_) { // 如果地图存在，则绘制地图点。
                DrawMapPoints();
            }

            pangolin::FinishFrame(); // 结束当前帧的绘制，并稍作延迟以控制帧率
            usleep(5000);
        }

        LOG(INFO) << "Stop viewer"; // 记录日志，表示Viewer已停止。
    }

    cv::Mat Viewer::PlotFrameImage() { // 用于绘制当前帧的图像
        cv::Mat img_out;
        cv:cvtColor( current_frame_->left_img_, img_out, cv::COLOR_GRAY2BGR ); // 将当前帧的左图像从灰度转换为BGR格式。左/右目的图像是归属于当前帧的，它的位姿为从立体视觉->左/右目
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) { // 循环遍历当前帧的左图像中的特征点
            if ( current_frame_->features_left_[i]->map_point_.lock() ) { // 如果特征点关联到了地图点，则在图像上绘制一个绿色的圆。 
            // map_point_ 是 feature 结构体中的一个 std::weak_ptr 的实例。如果原std::shared_ptr 仍然存在，.lock()将返回一个非空的std::shared_ptr；否则返回一个空的std::shared_ptr。
                auto feat = current_frame_->features_left_[i];
                cv::circle( img_out, feat->position_.pt, 2, cv::Scalar(0, 250, 0), 2);
            }
        }
        return img_out;
    }

    void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera ) { // 用于让相机跟随当前帧
        // 计算当前帧到相机的变换矩阵，并设置相机跟随该变换
        SE3 Twc = current_frame_->Pose().inverse(); // （current_frame_ 返回的 Pose 是 Tcw 世界系 -> 当前帧）
        pangolin::OpenGlMatrix m( Twc.matrix() );
        vis_camera.Follow( m, true );
    }

    void Viewer::DrawFrame( Frame::Ptr frame, const float* color ) { // 用于绘制帧的边界
        SE3 Twc = frame->Pose().inverse(); // 计算当前帧到世界系（或初始相机系）的变换矩阵
         
        const float sz = 1.0; // 矩形框的大小
        const int line_width = 2.0; // 绘制矩形框时，线条的宽度
        const float fx = 400; // 相机的焦距
        const float fy = 400;
        const float cx = 512; // 相机的光心坐标
        const float cy = 384;
        const float width = 1080; // 图像的宽度和高度（以像素为单位）
        const float height = 768;

        glPushMatrix(); // 保存当前的模型视图矩阵

        Sophus::Matrix4f m = Twc.matrix().template cast<float>(); // 将Sophus的SE3变换矩阵转换为Matrix4f类型（浮点数），以便与OpenGL兼容。
        glMultMatrixf( (GLfloat*)m.data() ); // 将当前模型视图矩阵与变换矩阵 m 相乘，以更新 OpenGL 的当前模型视图矩阵。

        if ( color == nullptr ) { // 如果未提供颜色，则使用默认颜色（红色）
            glColor3f(1, 0, 0);
        } else // 如果提供了颜色，则使用该颜色
            glColor3f( color[0], color[1], color[2] );

        glLineWidth( line_width ); // 设置绘制线条的宽度
        glBegin( GL_LINES ); // 开始绘制线条
        // 接下来定义量矩形框的8个顶点，并通过 glVertex3f() 绘制连接这些顶点的线条，以形成矩形框。
        // 这些顶点根据图像的宽度、高度、光心坐标和焦距计算得出，通过Twc变换转换为3D空间中的坐标。矩形框的绘制通过连接8个顶点形成的12条边（棱锥）来实现。
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz );

        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd(); // 结束线条的绘制。
        glPopMatrix(); // 恢复之前保存的模型视图矩阵。
    }

    void Viewer::DrawMapPoints() { // 用于绘制地图点
        const float red[3] = {1.0, 0, 0};
        for (auto& kf : active_keyframes_ ) { // 遍历活跃的关键帧，并以红色绘制它们。
            DrawFrame(kf.second, red);
            // 调用DrawFrame函数，将当前遍历到的关键帧（kf.second）和之前定义的红色数组red作为参数传递。这个函数根据给定的颜色和关键帧的位置信息在屏幕上绘制关键帧的框架。
        }

        glPointSize(2); // 设置接下来绘制的点的大小为2个单位。
        glBegin(GL_POINTS); // 传入GL_POINTS参数，表示开始绘制点集
        for (auto& landmark : active_landmarks_ ) { // 遍历active_landmarks_容器中的所有元素。active_landmarks_存储活跃地标。
            auto pos = landmark.second->Pos(); // 获取路标点位置信息
            glColor3f(red[0], red[1], red[2] ); // 设置接下来绘制的图形的颜色。这里使用之前定义的红色数组red作为参数。
            glVertex3d(pos[0], pos[1], pos[2] ); // 指定一个点的位置。这个函数在glBegin(GL_POINTS)和glEnd()之间调用，意味着在屏幕上绘制这个点
        }
        glEnd(); // 结束点集的绘制。所有在glBegin(GL_POINTS)和glEnd()之间指定的点将被绘制。
    }
} // namespace myslam