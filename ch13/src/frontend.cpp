#include <opencv2/opencv.hpp>
#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {
    Frontend::Frontend() { // 初始化GFTT特征点检测器、特征点数量等
        gftt_ = cv::GFTTDetector::create( Config::Get<int>("num_features"), 0.01, 20 ); // 创建GFTT特征点检测器。输入参数：要返回的最大特征点数量(int)、接受的特征点的最低质量水平(double)、两个特征点间的最小欧氏距离(int)。
        // 从Config的.yaml文件中，读取特征点的数量
        num_features_init_ = Config::Get<int>("num_features_init" ); 
        num_features_ = Config::Get<int>( "num_features" );
    }

    bool Frontend::AddFrame( myslam::Frame::Ptr frame ) { // 将新帧（图像）添加到前端处理流程中
        current_frame_ = frame; // 在这里检查frame->Pose()时，似乎就已经存在单位SE(3)了。看了下Frame的实例构建，似乎SE3 pose_;定义时就已默认赋单位阵，可能是新版本的特性？

        switch (status_ ) { // 根据前端的状态（初始化中、跟踪良好、跟踪不良、丢失），调用不同的处理函数。在Track()里面，前端的状态根据匹配的特征点数调整。
            case FrontendStatus::INITING:
                std::cout << "当前前端状态为 INITING " << std::endl;
                StereoInit(); // 初始化立体视觉的函数
                break;
            case FrontendStatus::TRACKING_GOOD:
                std::cout << "当前前端状态为 TRACKING_GOOD " << std::endl;
            case FrontendStatus::TRACKING_BAD:
                std::cout << "当前前端状态为 TRACKING_BAD " << std::endl;
                Track(); // 跟踪
                break;
            case FrontendStatus::LOST:
                std::cout << "当前前端状态为 LOST " << std::endl;
                Reset(); // 重置
                break;
        }

        last_frame_ = current_frame_;
        return true;
    }

    bool Frontend::Track() { // 跟踪：在当前帧和上一帧之间进行特征点匹配和位姿估计
        // current_frame_ 在 AddFrame() 插入帧的函数里就已经设置了。
        if (last_frame_) { // 根据上一帧的位姿，预测当前帧的位姿。
            current_frame_->SetPose( relative_motion_ * last_frame_->Pose() ); // relative_motion_ 为当前帧与上一帧的相对运动（上一次）。后面会更新这一次的相对运动。
        }

        int num_track_last = TrackLastFrame(); // 跟踪上一帧的特征点到当前帧
        tracking_inliers_ = EstimateCurrentPose(); // 估计当前帧的位姿（通过g2o图优化求解）。

        if (tracking_inliers_ > num_features_tracking_ ) { // 根据内点数 inliers（即匹配良好的特征点数量），更新前端状态（跟踪良好、跟踪不良、丢失）
            // tracking good
            status_ = FrontendStatus::TRACKING_GOOD;
        } else if (tracking_inliers_ > num_features_tracking_bad_ ) {
            // tracking bad
            status_ = FrontendStatus::TRACKING_BAD;
        } else {
            // lost
            status_ = FrontendStatus::LOST;
        }

        InsertKeyframe(); // 插入关键帧。是否关键的判定在该函数内部，通过成员变量“特征点数量”判定。
        relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse(); // 更新上一帧到当前帧的相对运动 

        if (viewer_) viewer_->AddCurrentFrame(current_frame_); // 更新 viewer
        // 尽管 frontend.h 和 viewer.h 里均有 Frame::Ptr current_frame_ = nullptr，但二者的current_frame_并不一样，尤其他们都是private的，作用域只在各自类体内
        return true;
    }

    bool Frontend::InsertKeyframe() { // 插入新的关键帧
        if (tracking_inliers_ >= num_features_needed_for_keyframe_ ) { // 判断是否为关键帧
            // still have enough features, don't insert keyframe
            return false;
        }
        // current frame is a new keyframe
        current_frame_->SetKeyFrame(); // 设置该帧为关键帧。包括关键帧标志位、ID
        map_->InsertKeyFrame( current_frame_ ); // map类的InsertKeyFrame()，不是Forntend的。在地图中插入新的关键帧

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "  << current_frame_->keyframe_id_;

        // 判定当前帧为关键帧的处理：提取新特征点、寻找左右目匹配的特征点并三角化建立新路标、新关键帧和路标点加入地图（触发一次后端优化）
        SetObservationsForKeyFrame(); // 关键帧中的特征点设置地图点观察。通过lock检测特征点是否关联地图点，若有关联则增加一个特征观测（观测维度）
        DetectFeatures(); // detect new features
        FindFeaturesInRight(); // track in right image
        TriangulateNewPoints(); // triangulate map points （对左右目匹配的特征点进行三角化）
        std::cout << "> 判定为关键帧。运行backend_->UpdateMap(),以触发后端优化" << std::endl;
        backend_->UpdateMap(); // update backend because we have a new keyframe
        std::cout << ">> 退出Backend::UpdateMap(),并销毁Backend::UpdateMap()内的lock(data_mutex_) (到此即运行Backend::BackendLoop()的map_update_.wait(lock)往后)" << std::endl;

        if (viewer_) viewer_->UpdateMap(); // 如果开启了viewr，则更新其视图
        return true;
    }

    void Frontend::SetObservationsForKeyFrame() {
        for (auto &feat : current_frame_->features_left_) {
            auto mp = feat->map_point_.lock(); // Frontend类的current_frame_，相应的Frame类的Feature智能指针左目特征features_left_，有无Feature类的与MapPoint类关联的weak_ptr地图点 map_point_
            if (mp) mp->AddObservation(feat); // 若有关联的地图点，则将该特征观测加入观测列表（增加观测变量维度）
        }
    }
    
    int Frontend::TriangulateNewPoints() { // 通过左右目新匹配的特征点，三角化新的路标点
        std::vector<SE3> poses{ camera_left_->pose(), camera_right_->pose() }; // 获取左右目相机的位姿。该pose()为camera类的，立体视觉->左/右目。
        SE3 current_pose_Twc = current_frame_->Pose().inverse(); // 当前帧->世界系的位姿。当前帧的Pose()为Frame类的。
        int cnt_triangulated_pts = 0; // 三角化的点数
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i ) { // 遍历当前帧左目的特征点。current_frame_是Frontend类的Frame指针，features_left_是Frame类的Feature容器。
            if (current_frame_->features_left_[i]->map_point_.expired() &&  // map_point_是Feature类指向MapPoint的weak_ptr // .expired()函数用于检查weak_ptr是否过期，如未指向有效的shared_ptr则返回true。
                current_frame_->features_right_[i] != nullptr ) {
                // 左图的特征点未关联地图点，且存在右图匹配点，尝试三角化
                std::vector<Vec3> points{ // 分别存入当前帧中，左右目匹配的特征点的位置
                    camera_left_->pixel2camera( Vec2( current_frame_->features_left_[i]->position_.pt.x, 
                                                    current_frame_->features_left_[i]->position_.pt.y )   ), // pixel2camera按理还要有第二个double型参数作为深度输入。这里为什么没有？
                    camera_right_->pixel2camera( Vec2( current_frame_->features_right_[i]->position_.pt.x,
                                                    current_frame_->features_right_[i]->position_.pt.y )  )
                };
                Vec3 pworld = Vec3::Zero(); // 初始化三维路标点

                if ( triangulation(poses, points, pworld) && pworld[2] > 0 ) { // 调用triangulation()进行三角化。若三角化成功，且点在世界系的z坐标>0（即点在相机前方），则创建一个新的地图点，并将其添加到地图中。
                    auto new_map_point = MapPoint::CreateNewMappoint(); // 工厂模式创建地图点实例。
                    pworld = current_pose_Twc * pworld; // 三维路标点转换至世界系
                    new_map_point->SetPos( pworld ) ;
                    new_map_point->AddObservation( current_frame_->features_left_[i] ) ;
                    new_map_point->AddObservation( current_frame_->features_right_[i] );

                    current_frame_->features_left_[i]->map_point_ = new_map_point;
                    current_frame_->features_right_[i]->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    int Frontend::EstimateCurrentPose() { // 通过g2o库，估计当前帧的相机位姿
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType; // g2o::BlockSolver 管理求解过程的优化变量(6维相机位姿)和中间量(3维路标点)。
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 指定线性求解器
        auto solver = new g2o::OptimizationAlgorithmLevenberg(  // 梯度下降方法为LM
                                std::make_unique<BlockSolverType>(   std::make_unique<LinearSolverType>()   )    );
        g2o::SparseOptimizer optimizer; // g2o::SparseOptimizer 稀疏优化器，管理优化过程
        optimizer.setAlgorithm(solver); // 设置求解器

        // vertex
        VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose. 创建相机顶点
        vertex_pose->setId(0); 
        vertex_pose->setEstimate( current_frame_->Pose() ); // 顶点初始值设置为当前帧的位姿。初始位姿通过上一次的帧与帧间相对运动，以及上一帧的位姿，进行估计。
        optimizer.addVertex( vertex_pose );

        // K
        Mat33 K = camera_left_->K(); // 当前帧的内参，即左目的内参？

        // edges // ***
        // 此处的边已经匹配了特征点关联的地图点，只在当前帧上进行操作，不比较上一帧的灰度值。之前先通过LK光流TrackLastFrame()追踪了上一帧到这一帧的特征点。 
        // 在TrackLastFrame()利用LK光流追踪后，还将当前帧的特征点与已有的三维地图点进行关联（通过赋予与上一帧特征点相同的关联）
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges; // 一元边（仅连接相机位姿顶点）的容器
        std::vector<Feature::Ptr> features; // 特征指针的容器。
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i ) { // 遍历当前帧的左目特征
            auto mp = current_frame_->features_left_[i]->map_point_.lock(); // 锁定每个特征点关联的地图点。
            if (mp) { // 如果存在地图关联
                features.push_back( current_frame_->features_left_[i] ); // 存入刚刚创建的特征点容器。里面放的是有地图点关联的左目特征。
                EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly( mp->pos_, K ); // 新建一元边。传入左目特征点对应的三维路标点位置（世界系）、左目相机内参
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                edge->setMeasurement( 
                            toVec2( current_frame_->features_left_[i]->position_.pt )   ); // 设置边的估计值，为左目特征点的像素坐标。
                edge->setInformation( Eigen::Matrix2d::Identity() ); // 信息矩阵（噪声协方差的逆），没有就按单位矩阵
                edge->setRobustKernel( new g2o::RobustKernelHuber ); // 鲁棒核函数，设置为Huber核
                edges.push_back(edge); // 一元边存入容器。放入的是左目特征的观测
                optimizer.addEdge(edge);
                index++;
            }
        }

        // estimate the Pose the determine the outliers
        const double chi2_th = 5.991; // 卡方阈值，用于判断观测值与估计值的差异是否过大，从而判定是否异常。这个值对应95%置信水平下的卡方分布值（自由度为2）
        int cnt_outlier = 0; // 异常特征点的计数
        for (int iteration = 0; iteration < 4; ++iteration) { // 迭代4次，通过多次迭代优化，逐步逼近最优解。4次为经验值。
            vertex_pose->setEstimate( current_frame_->Pose() ); // 每次迭代都重新设置当前帧的位姿
            optimizer.initializeOptimization(); // 初始化优化过程
            optimizer.optimize(10); // 每次迭代的优化次数是10次。
            cnt_outlier = 0; // 每次迭代都重置异常点计数

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i ) { // 遍历有地图点关联的左目特征
                auto e = edges[i];
                if (features[i]->is_outlier_) { // 判定该左目特征是否为异常点。若是异常点，执行下面代码。
                    e->computeError(); // 一元边的误差计算函数：左目特征点的像素坐标，减去由三维路标点位置和相机位姿估计得到的估计像素坐标。
                }

                if (e->chi2() > chi2_th ) { // chi2() 通常用于计算误差的平方和（也称为卡方统计量）
                    features[i]->is_outlier_ = true; // 该左目特征设置为异常点
                    e->setLevel(1); // setLevel() 设置边的层级，用于控制优化过程中的处理优先级。优先级取决于优化器的定义。
                    cnt_outlier++; // 异常的左目特征计数+1
                } else {
                    features[i]->is_outlier_ = false; // 不是异常点。（逻辑上，是先对异常点进行优化）
                    e->setLevel(0); 
                }

                if (iteration == 2) { // 迭代次数3次后，去除Huber核函数。使用Huber核函数可以减小异常点对优化结果的影响，提高系统的鲁棒性。
                // 在迭代初期，异常点较多，使用Huber核函数有助于稳定优化过程；而在迭代后期，异常点已经得到较好的处理或剔除，此时使用更精确的误差项进行优化可能更为有利。
                    e->setRobustKernel( nullptr );
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimationg: " << cnt_outlier << " / " << features.size() - cnt_outlier;
        // Set pose and outlier
        current_frame_->SetPose( vertex_pose->estimate() ); // 最终迭代结果的位姿估计
        LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

        for (auto &feat : features) {
            if (feat->is_outlier_) { // 迭代后仍是异常点的
                feat->map_point_.reset(); // shared_ptr的.reset()用于重置shared_ptr，使其不再管理之前的对象。如果它是最后一个拥有该对象的shared_ptr，则释放该对象（地图点）。   
                feat->is_outlier_ = false; // maybe we can still use it in future. 即使特征在当前迭代中被认为是异常点，它仍可能包含有关场景或相机运动的有用信息。保留这些信息可能有助于在未来改进系统的性能或准确性。
            }
        }
        return features.size() - cnt_outlier; // 返回有效左目特征点的数量
    }

    int Frontend::TrackLastFrame() { // 使用LK光流法，追踪上一帧的特征点到当前帧
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame_->features_left_) { // 遍历上一帧的左目特征点。last_frame_ 为 Frontend类 中指向 Frame 的 shared_ptr；features_left_ 为 Frame类 中存储 Feature类shared_ptr的容器。
            if (kp->map_point_.lock() ) { // 如果特征点有对应的地图点时。map_point_ 为 Feature类 中指向 MapPoint 的 weak_ptr；MapPoint 的实际 shared_ptr 归 Map类 持有。
                // use project point
                auto mp = kp->map_point_.lock(); // 将关联的地图点赋值
                auto px = camera_left_->world2pixel( mp->pos_, current_frame_->Pose() ); // 关联的三维地图点，世界系->当前帧的左目像素坐标系
                kps_last.push_back(kp->position_.pt); // 存储上一帧的左目特征点位置
                kps_current.push_back(  cv::Point2f(px[0], px[1] )  ); // 当前帧左目特征点位置的初始猜测
            } else { // 无对应地图点时，依然存放特征点位置，作为LK光流追踪的初始猜测。
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(   // 用于计算稀疏光流的函数，它实现了Lucas-Kanade方法在图像金字塔上的扩展
            last_frame_->left_img_, current_frame_->left_img_, kps_last, kps_current, // 前一帧和当前帧的图像（光流通常在灰度图像上计算），前一帧和(输出)当前帧图像的特征点
            status, error, cv::Size(11, 11), 3,  // status(输出)数组，特征点有效匹配则对应元素为1。error(输出)数组，每个特征点跟踪的误差估计。cv::Size(11, 11) 搜索窗口大小(像素)，在每个金字塔层级上搜索特征点的对应点。3 是金字塔的最大层级数。
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01), // 迭代搜索算法的终止条件，此处设置为达到30次迭代或每次迭代的变化小于0.01时停止。cv::TermCriteria::COUNT 表示迭代次数，cv::TermCriteria::EPS 表示迭代变化量的阈值。
            cv::OPTFLOW_USE_INITIAL_FLOW  ); // cv::OPTFLOW_USE_INITIAL_FLOW 为一个标志，指示算法是否应使用前一个光流估计作为当前迭代的初始估计。
        
        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i ) {
            if (status[i]) { // 对于成功追踪的特征点，创建新的特征点对象，并将其添加到当前帧的特征点列表中。status[i] 如果为true，则表示第i个特征点成功被跟踪。
                cv::KeyPoint kp(kps_current[i], 7); // 创建一个新的cv::KeyPoint对象。kps_current[i]是当前帧中成功跟踪的特征点的位置。7被传递给cv::KeyPoint的构造函数作为该特征点的尺度（size）。
                Feature::Ptr feature(new Feature(current_frame_, kp) ); // 创建一个新的Feature对象。构造函数接受当前帧的引用和刚刚创建的特征点kp作为参数。
                feature->map_point_ = last_frame_->features_left_[i]->map_point_; // 将新创建的特征点的map_point_成员设置为与前一帧中对应特征点的map_point_相同的值。此处关联了当前帧中特征点的地图点
                current_frame_->features_left_.push_back(feature); // 将新创建的特征点(ptr)，添加到当前帧的features_left_列表
                num_good_pts++;
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;
    }

    bool Frontend::StereoInit() { // 状态初始化函数
        int num_features_left = DetectFeatures(); // 检测当前帧左图向的特征点并存储，返回检测到的特征点数量。
        int num_coor_features = FindFeaturesInRight(); // 通过LK光流，寻找与左目特征点对应的右目特征点，并返回正确匹配的右目特征点的数量。
        if (num_coor_features < num_features_init_) { // 如果右目正确特征点的数量小于正确初始化设定的边界，则视为初始化失败。
            return false;
        }

        bool build_map_success = BuildInitMap(); // 初始化地图。若初始化成功，则返回true
        if (build_map_success) {
            status_ = FrontendStatus::TRACKING_GOOD; // 修改前端状态为正确追踪
            if (viewer_) { // 如果启用了viewr，添加当前帧并更新地图
                viewer_->AddCurrentFrame(current_frame_);
                viewer_->UpdateMap();
            }
            return true;
        }
        return false;
    }

    int Frontend::DetectFeatures() {
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255); // 创建一个cv::Mat对象mask 作为掩码图像使用。
        // 掩码图像的大小与当前帧的左图像（current_frame_->left_img_）相同。CV_8UC1指定了图像的数据类型，即8位无符号单通道（灰度图）。初始值255表示掩码图像的所有像素都被设置为白色。
        for (auto &feat : current_frame_->features_left_ ) { // 遍历当前帧的左目特征点。初始化时还没有特征点，则会跳过。
            cv::rectangle( mask, feat->position_.pt - cv::Point2f(10, 10),  // 在掩码图像mask上绘制一个矩形。
                            feat->position_.pt + cv::Point2f(10, 10), 0, -1); // 新版中，最后的“-1”表示填充（同旧版CV_FILLED）
            // 矩形的左上角坐标为特征点位置偏移-10像素，右下角坐标是特征点位置偏移+10像素。矩形的颜色设置为0（黑色），-1表示矩形内部应该被填充。
        }

        std::vector<cv::KeyPoint> keypoints; // 用于存储检测到的特征点
        gftt_->detect(current_frame_->left_img_, keypoints, mask); // 调用gftt_对象的detect方法，在当前帧的左图像上检测特征点。
        // 检测到的特征点被添加到keypoints向量。使用前面创建的掩码图像mask来排除已经存在特征点的区域。
        int cnt_detected = 0; // 用于计数检测到的特征点数量
        for (auto &kp : keypoints) {  // 遍历检测到的特征点列表
            current_frame_->features_left_.push_back(   Feature::Ptr( new Feature(current_frame_, kp) )   ); 
            // 为每个检测到的特征点创建一个新的Feature对象，并将其添加到current_frame_的左图像特征点列表 features_left_
            cnt_detected++;
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected; // 返回检测到的特征点数量。检查点
    }

    int Frontend::FindFeaturesInRight() { // 通过LK光流，寻找与左目特征点对应的右目特征点，并返回正确匹配的右目特征点的数量。
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_left, kps_right; // 分别存储左图像和右图像中的特征点位置
        for (auto &kp : current_frame_->features_left_) { // 遍历当前帧的左目特征点
            kps_left.push_back(kp->position_.pt);
            auto mp = kp->map_point_.lock();
            if (mp) { // 如果存在与当前左目特征点关联的地图点
                // use projected points as initial guess
                auto px = camera_right_->world2pixel( mp->pos_, current_frame_->Pose() ); // 三维路标点，世界系->当前帧右目像素坐标
                kps_right.push_back( cv::Point2f(px[0], px[1]) ); // 存储右目特征点的像素坐标
            } else {
                // use same pixel in left image
                kps_right.push_back(kp->position_.pt); // 如果不存在关联地图点，则往右目关键点容器存入左目特征点的像素位置，作为初始猜测。
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(  // LK光流追踪。和上一个函数的不同点在于，上一个是从上一帧到当前帧，这里是从左目到右目
            current_frame_->left_img_, current_frame_->right_img_, kps_left, kps_right, // kps_right前面存入特征点位置可作为初始猜测，非必需但有助于正确收敛。
            status, error, cv::Size(11, 11), 3, 
            cv::TermCriteria( cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01 ),
            cv::OPTFLOW_USE_INITIAL_FLOW     );
        
        int num_good_pts = 0; // 用于计数在右图像中找到的有效特征点数量
        for (size_t i = 0; i < status.size(); ++i ) { 
            if (status[i]) { // 检查每个特征点的跟踪状态。
                cv::KeyPoint kp(kps_right[i], 7 ); // 创建一个新的cv::KeyPoint对象。kps_current[i]是当前帧中成功跟踪的特征点的位置。7被传递给cv::KeyPoint的构造函数作为该特征点的尺度（size）。
                Feature::Ptr feat(new Feature( current_frame_, kp )  ); // 创建一个新的Feature对象。构造函数接受当前帧的引用和刚刚创建的特征点kp作为参数。
                feat->is_on_left_image_ = false; // 表示这个特征点是在右图像中找到的
                current_frame_->features_right_.push_back(feat);
                num_good_pts++;
            } else {
                current_frame_->features_right_.push_back(nullptr); // 特征点跟踪失败时，添加空指针（nullptr）而非移去，可能是为了正确的索引左右目特征点一一对应的关系。
            }
        }
        LOG(INFO) << "Find " << num_good_pts << " in the right image. ";
        return num_good_pts; // 检查点
    }

    bool Frontend::BuildInitMap() { // 初始化地图。包括三角化当前帧左右目的路标点、将路标点加入地图并与当前帧的特征点关联、同时设置当前帧为关键帧（因为是初始地图）
        std::vector<SE3> poses{ camera_left_->pose(), camera_right_->pose() }; // 分别为立体视觉->左/右目。这里的pos()是在Dataset::Init()读取calib.txt的相机参数时赋值的。
        size_t cnt_init_landmarks = 0; // 用于记录成功创建的地图点数量 
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i ) { // 遍历当前帧的左目特征点  
            if (current_frame_->features_right_[i] == nullptr )  continue; // 如果对应的右目特征点不存在（即为nullptr），则跳过当前循环迭代 
            // create map point from triangulation
            std::vector<Vec3> points{
                camera_left_->pixel2camera( // 左目相机系。没有显式地加入第二个double型的深度输入参数以得到相机归一化坐标，后面再三角化计算深度。
                    Vec2( current_frame_->features_left_[i]->position_.pt.x,
                        current_frame_->features_left_[i]->position_.pt.y)   ),
                camera_right_->pixel2camera( // 右目相机系
                    Vec2( current_frame_->features_right_[i]->position_.pt.x,
                        current_frame_->features_right_[i]->position_.pt.y)  )
            };
            Vec3 pworld = Vec3::Zero(); // 用于存储三角化结果的三维世界坐标点  

            if (triangulation(poses, points, pworld) && pworld[2] > 0 ) { // 如果三角化成功且点的z坐标大于0（表示点在相机前方）
                auto new_map_point = MapPoint::CreateNewMappoint(); // 工厂模式创建新的地图点实例
                new_map_point->SetPos(pworld); // 设置地图点的世界坐标  
                new_map_point->AddObservation(current_frame_->features_left_[i]); // 将当前特征点（左目和右目）作为观测添加到地图点中  
                new_map_point->AddObservation(current_frame_->features_right_[i]);
                current_frame_->features_left_[i]->map_point_ = new_map_point; // 将地图点的指针设置到对应的特征点中，表示这些特征点已经找到了它们对应的地图点  
                current_frame_->features_right_[i]->map_point_ = new_map_point; 
                cnt_init_landmarks++; // 增加成功创建的地标点计数
                map_->InsertMapPoint(new_map_point); // 将新的地图点插入到地图中
            }
        }
        current_frame_->SetKeyFrame(); // 将当前帧设置为关键帧。初始化地图时，可以直接设置为关键帧。
        map_->InsertKeyFrame(current_frame_);
        backend_->UpdateMap(); // 通知后端更新地图（可能包括优化、滤波等操作）  

        LOG(INFO) << "Initial map created with " << cnt_init_landmarks << " map points"; // 检查点
        return true;
    }

    bool Frontend::Reset() {
        LOG(INFO) << "Reset is not implemented. 跟踪丢失、重置。(没有代码实现,只简单返回true) ";
        return true;
    }

} // namespace myslam