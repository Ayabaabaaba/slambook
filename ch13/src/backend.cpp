#include "myslam/backend.h" 
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {
    Backend::Backend() { // 构造函数。
        backend_running_.store(true); // 初始化backend_running_原子布尔变量为true，表示后端正在运行
        backend_thread_ = std::thread( std::bind(&Backend::BackendLoop, this)  ); // 创建一个新线程来运行BackendLoop成员函数（后端的主要循环）。
        // this 是类体内的隐式指针，用于指向调用成员函数的实例对象。
    }

    void Backend::UpdateMap() { // 通知后端处理新数据（以便更新地图）
        std::cout << ">> 进入Backend::UpdateMap(),即将锁定互斥锁 lock(data_mutex_)" << std::endl; // 一旦判别为关键帧，即运行至此处，而下一行的lock(data_mutex_)，会等待Backend::BackendLoop()运行完lock(data_mutex_)后才运行（前端毕竟比后端快）。
        // 每次运行都可能在不同帧的位置上，在此处报错。感觉需要在数据结构等了解互斥锁的工作原理后，才能解决。
        std::unique_lock<std::mutex> lock(data_mutex_); // 获取一个互斥锁。进入 Backend::BackendLoop() 的 map_update_.wait(lock) 线程。目前来看，每次运行时，会随机在不同帧的这行互斥锁里报错，原因估计为还没运行到Backend::BackendLoop()的lock(data_mutex_)，导致找不到互斥锁？（即使暂停主线程，backend_线程也不运行。解决方法未知）
        std::cout << ">> Backend::UpdateMap() 完成 lock(data_mutex_) " << std::endl;
        map_update_.notify_one(); // .notify_one() 会解除一个正在等待该条件变量的线程的阻塞状态。退出Backend::UpdateMap()后会通知Backend::BackendLoop()的map_update_.wait(lock)
        std::cout << ">> 完成 map_update_.notify_one(),即将退出 Backend::UpdateMap()" << std::endl;
    }

    void Backend::Stop() {
        backend_running_.store(false); // 设置原子bool量为false，通知BackendLoop循环应该退出
        map_update_.notify_one(); // 再次调用.notify_one()来确保循环能够尽快检查到停止信号（尽管可能不是必需的，因为循环本身也会检查backend_running_）（还不太理解逻辑？）
        backend_thread_.join(); // 调用.join()来等待后端线程结束。.
        // .join()函数在多线程编程中用于等待一个线程终止。调用线程（通常是主线程）会被阻塞，直到被调用的线程（此处为backend_thread_）完成执行。这个机制确保了线程之间的同步。
        std::cout << ">>>>> 后端线程停止,整个VO停止 " << std::endl;
    }

    void Backend::BackendLoop() {  // 后端的主循环
        while (backend_running_.load() ) { // 不断检查backend_running_变量
        // .load 读取 std::atomic<> 实例的值。对于std::atomic<> 的原子操作，读取（.load()）与写操作（.store()或=）是不可分割的，可以安全地在多线程环境使用。原子类型的读写通常比互斥锁更快，因为不需要上下文切换或阻塞等待。
        // 当UpdateMap函数调用notify_one时，这里会被唤醒。
            std::unique_lock<std::mutex> lock(data_mutex_); // 获取互斥锁。
            std::cout << ">>>> Backend::BackendLoop() 锁定lock(data_mutex_),无限循环等待 map_update_.wait(lock)" << std::endl; // 在初始化backend_实例后即进入此处待机。
            map_update_.wait(lock); // 等待map_update_条件变量。此处会先释放Backend::BackendLoop()的lock(data_mutex_)，并等待接收到 lock(data_mutex_)的通知。那么前面的报错，就是这里没能成功释放锁？为什么会是Segmentation fault？
            std::cout << ">>> 获知 map_update_.wait(lock),进入 Backend::BackendLoop()" << std::endl;

            // 后端仅优化激活的Frames和Landmarks
            Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
            Optimize(active_kfs, active_landmarks); // 触发一次后端优化
            std::cout << ">>>> Backend::BackendLoop() 完成触发的一次后端优化" << std::endl;
        }
    }

    void Backend::Optimize(Map::KeyframesType &keyframes,  Map::LandmarksType &landmarks) { // 后端优化的核心，接受关键帧和路标的集合作为输入。
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType; // g2o::BlockSolver 管理求解过程的优化变量(6维相机参位姿)和中间量(3维路标点)。
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType; // 指定线性求解器为 CSparse 库（一个稀疏矩阵求解器）
        auto solver = new g2o::OptimizationAlgorithmLevenberg(  // 梯度下降方法为LM
                std::make_unique<BlockSolverType>( std::make_unique<LinearSolverType>() )   );
        g2o::SparseOptimizer optimizer; // 图模型
        optimizer.setAlgorithm(solver); // 设置求解器

        // pose 顶点，使用keyframe id
        std::map<unsigned long, VertexPose *> vertices; // std::map提供了基于键（此处为unsigned long类型的keyframe id）的快速查找功能。std::map中的元素是按键排序的，不允许键重复。
        unsigned long max_kf_id = 0; // 意义？
        for (auto &keyframe : keyframes) { 
            auto kf = keyframe.second; // .first是其ID，.second是帧结构体。
            VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose. 相机位姿顶点
            vertex_pose->setId(kf->keyframe_id_); // 设置顶点的id为关键帧的id
            vertex_pose->setEstimate(kf->Pose() ); // 设置顶点的估计值为关键帧的位姿
            optimizer.addVertex(vertex_pose); 
            if (kf->keyframe_id_ > max_kf_id) {   // 更新最大的keyframe id
                max_kf_id = kf->keyframe_id_;
            }

            vertices.insert( {kf->keyframe_id_, vertex_pose} ); // // 将顶点添加到std::map中，使用关键帧id作为键。
        }

        std::map<unsigned long, VertexXYZ *> vertices_landmarks; // 初始化存储路标顶点的std::map，使用路标id作为键

        // K 和左右外参
        Mat33 K = cam_left_->K();
        SE3 left_ext = cam_left_->pose(); // 左右目的pose()，是立体视觉（可以理解为当前帧？）->左/右目
        SE3 right_ext = cam_right_->pose();

        // edges
        int index = 1; // 边的id，从1开始计数
        double chi2_th = 5.991; // robust kernel 阈值
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features; // 存储边和对应特征的std::map

        for (auto &landmark : landmarks) { // 遍历所有路标
            if (landmark.second->is_outlier_)  continue; // 如果路标是外点，则跳过
            unsigned long landmark_id = landmark.second->id_; // 路标ID
            auto observations = landmark.second->GetObs(); // 获取所有观测到这个路标点的特征点的weak_ptr列表
            for (auto &obs : observations) {
                if (obs.lock() == nullptr) continue;  // 如果观测值是空的，则跳过
                auto feat = obs.lock(); // 获取观测值对应的特征
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr)  continue;  // 如果特征是外点或所属帧是空的，则跳过
                auto frame = feat->frame_.lock(); // 获取特征所属的关键帧
                EdgeProjection *edge = nullptr; // 初始化边指针
                if (feat->is_on_left_image_) {
                    edge = new EdgeProjection(K, left_ext); // 创建一个新的投影边(二元边)，使用左相机的内参和外参
                } else {
                    edge = new EdgeProjection(K, right_ext); // 创建一个新的投影边，使用右相机的内参和外参
                }

                // 如果landmark还没有被加入优化，则新加一个顶点
                if (vertices_landmarks.find(landmark_id) == vertices_landmarks.end() ) { 
                //.find()在容器中搜索具有指定键(landmark_id)的元素，如果找到了则返回指向该元素的迭代器，如果没找到则返回指向容器末尾的迭代器。
                    VertexXYZ *v = new VertexXYZ; // 创建新的路标顶点
                    v->setEstimate(landmark.second->Pos() ); // 设置顶点的估计值为路标的位置
                    v->setId(landmark_id + max_kf_id + 1 ); // 设置顶点的id，这里使用了一个偏移量来避免与关键帧id冲突。
                    v->setMarginalized(true); // 设置顶点为被边缘化的状态
                    vertices_landmarks.insert( {landmark_id, v} ); // 将顶点添加到std::map
                    optimizer.addVertex(v); // 将顶点添加到优化器
                }

                // 如果相机位姿顶点和路标顶点都存在，则创建并添加一条边。
                if (vertices.find(frame->keyframe_id_) != vertices.end() 
                    && vertices_landmarks.find(landmark_id) != vertices_landmarks.end() ) {
                    
                    edge->setId(index); // 设置边的id
                    edge->setVertex(0, vertices.at(frame->keyframe_id_) ); // pose
                    edge->setVertex(1, vertices_landmarks.at(landmark_id) );  // landmark
                    edge->setMeasurement( toVec2(feat->position_.pt) ); // 设置边的测量值为特征的位置
                    edge->setInformation( Mat22::Identity() ); // 设置边的信息矩阵为单位矩阵
                    auto rk = new g2o::RobustKernelHuber(); // 创建一个Huber鲁棒核函数的实例
                    rk->setDelta(chi2_th); // 设置鲁棒核函数的阈值
                    edge->setRobustKernel(rk); // 为边设置鲁棒核函数
                    edges_and_features.insert( {edge, feat} ); // 将边和对应的特征添加到std::map中
                    optimizer.addEdge(edge);
                    index++;
                } else delete edge; // 如果边没有被添加，则释放其内存
            }
        }

        // do optimization and eliminate the outliers. 执行优化并剔除外点
        optimizer.initializeOptimization(); // 初始化优化过程。  
        optimizer.optimize(10); // 设置优化迭代的最大次数。这里即运行了优化

        // 优化结果处理
        int cnt_outlier = 0, cnt_inlier = 0; // 分别用于统计外点和内点的数量
        int iteration = 0;
        while (iteration < 5 ) { // 最多迭代5次
            cnt_outlier = 0;
            cnt_inlier = 0;
            // determine if we want to adjust the outlier threshold
            for (auto &ef : edges_and_features) { // 遍历所有的边和特征（edges_and_features）
                if (ef.first->chi2() > chi2_th ) { // 如果当前边（或特征）的卡方值（chi2）大于外点阈值（chi2_th），则认为它是一个外点
                    cnt_outlier++; // 统计外点数量。
                } else {
                    cnt_inlier++;
                }
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier ); // 计算内点比例，即内点数量除以总点数。
            if (inlier_ratio > 0.5) { // 如果内点比例大于0.5，即超过一半的数据被认为是内点，则认为当前的外点阈值合适，跳出循环。
                break;
            } else {
                chi2_th *= 2; // 将外点阈值加倍，以包含更多的点作为内点
                iteration++;
            }
        }

        for (auto &ef : edges_and_features ) { // 再次遍历所有的边和特征，这次是为了根据最终的外点阈值来标记外点和内点
            if (ef.first->chi2() > chi2_th) { // 如果当前边（或特征）的卡方值大于最终的外点阈值，则标记为外点。
                ef.second->is_outlier_ = true;
                // remove the observation
                ef.second->map_point_.lock()->RemoveObservation(ef.second);
            } else {
                ef.second->is_outlier_ = false;
            }
        }

        LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/" << cnt_inlier;

        // Set pose and landmark position
        for (auto &v : vertices) { // 遍历所有的相机位姿顶点
            keyframes.at(v.first)->SetPose(v.second->estimate() ); // 根据优化结果设置关键帧的位姿
        }
        for (auto &v : vertices_landmarks) {
            landmarks.at(v.first)->SetPos(v.second->estimate() ); // 根据优化结果设置路标的位置
        }
    }
    
} // namespace myslam