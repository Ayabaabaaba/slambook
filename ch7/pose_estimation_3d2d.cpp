#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

// 特征点匹配
void find_feature_matches( const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1, std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches );

// 像素坐标 -> 相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o. g2o库用于图优化。
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d; 
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
// std::vector 默认使用 allocator 不保证元素的内存对齐，可能导致访问的性能下降。aligned_allocator 为内存对齐

// BA求解，非线性用g2o图优化
void bundleAdjustmentG2O( 
    const VecVector3d &points_3d,   const VecVector2d &points_2d,
    const Mat &K,    Sophus::SE3d &pose 
);

// BA by gauss-newton. 此处未手写函数，也可用 Ceres 库的函数求解。
void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,   const VecVector2d &points_2d,
    const Mat &K,       Sophus::SE3d &pose
);


int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "ussage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    // 读取图像
    Mat img_1 = imread(argv[1], IMREAD_COLOR);
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    // 匹配特征点
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat d1 = imread(argv[3], IMREAD_UNCHANGED); // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 内参矩阵

    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for (DMatch m: matches) { 
        ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y)) [int(keypoints_1[m.queryIdx].pt.x)]; 
        // 从深度图d1中，获取关键点 keypoints_1[m.queryIdx] 位置的深度值。 
        // .ptr<unsigned shor>(int row) 是 OpenCV 中 Mat 类的一个成员函数，返回指向指定行开始位置的指针，指针类型是 unsigned short*（因为深度图使用16位无符号整数 ushort 存储深度值）
        // int(keypoints_1[m.queryIdx].pt.y 和 int(keypoints_1[m.queryIdx].pt.x) 分别将关键点的 y 和 x 坐标转换为整数，因为坐标是浮点数，行和列索引为整数。

        if (d == 0) continue; // bad depth
        float dd = d / 5000.0; // 将深度值 d（ushort）转换为浮点数 dd，5000.0 为缩放因子，与硬件相关的固有属性。
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K); // 像素坐标 -> 相机归一化
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd)); // 相机归一化 -> 3D世界坐标(预测值)
        pts_2d.push_back(keypoints_2[m.trainIdx].pt); // 2D像素投影坐标（观测值）
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    /*-----1. OpenCV 的 EPnP 求解---------*/
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r,t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false,SOLVEPNP_EPNP); // 调用 OpenCV 的PnP求解，可选EPnP、DLS等方法。
    // pts_3d（3D世界系中的点集），pts_2d（2D投影系中的点集），K（相机内参），distCoeffs（相机畸变系数，校准后可以留空），rvec（输出旋转向量），tvec（输出平移向量），useExtrinsicGuess（bool值，默认为false，指示是否使用rvec和tvec作为旋转和平移的初始估计），flags（指定PnP求解方法，默认为SLOVEPNP_ITERATIVE，还有SOLVEPNP_P3P、SOLVEPNP_EPNP等）
    Mat R;
    cv::Rodrigues(r, R); // r 为旋转向量，通过罗德里格斯公式转换为矩阵
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "Solve PnP in OpenCV cost time: " << time_used.count() << " seconds." << endl;
    cout << "R = " << endl << R << endl;
    cout << "t = " << endl << t << endl;

    /*------非线性优化：BA求解（只求解了相机位姿，没有优化特征点位置）---------*/
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z)); 
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }
    // 此处转换点集的类型，方便后续函数调用。
    // pts_3d 为 vector<Point3f> 类型；pts_3d_eigen 为 vector<Eigen::Vector3d> 类型，eigen类型便于使用eigen库求解。

    /*------2. BA求解，非线性最小二乘通过 Gauss-Newton 求解---------*/
    cout << endl << "Calling bundle adjustment by gauss newton " << endl;
    Sophus::SE3d pose_gn; // 在定义 Sophus::SE3d 类型的变量时，即赋予了单位矩阵初始值
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn); 
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << " Solve PnP by gauss newton cost time : " << time_used.count() << "seconds. " << endl;

    /*------3. BA求解，非线性最小二乘通过 g2o 图优化求解---------*/
    cout << endl << "Calling bundle adjustment by g2o " << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << " Solve PnP by g2o cost time : " << time_used.count() << " seconds " << endl;
    return 0;
}


// 匹配特征点
void find_feature_matches( const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1, std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches ){
    // 初始化
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    // 第一步：检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    // 第二步：根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    // 第三步：使用 Hamming 距离，匹配 BRIEF 描述子
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    // 第四步：匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; i++){
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

// 像素坐标 -> 相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

/*--------2. BA求解，非线性问题通过Gauss-Newton求解，此处为手写，也可调用Ceres库---------*/
void bundleAdjustmentGaussNewton( 
    const VecVector3d &points_3d, const VecVector2d &points_2d,
    const Mat &K, Sophus::SE3d &pose) 
    {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10; // 迭代次数
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0); // 内参
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++) { // 在for条件里。iter++ 未用于赋值，与 ++iter 效果一致。 
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero(); // Hessian矩阵
        Vector6d b = Vector6d::Zero(); // bias，增量方程右边的常数项。

        cost = 0;
        // compute cost
        for (int i = 0; i < points_3d.size(); i++) {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2]; 
            double inv_z2 = inv_z * inv_z; // 这两项用于代入 Jacobian矩阵的解析形式
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy); // 相机模型，从系1的3D位置得到的2D预测值

            Eigen::Vector2d e = points_2d[i] - proj; // cost function

            cost += e.squaredNorm(); // 范数的平方

            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,   0,    fx * pc[0] * inv_z2,    // 雅克比矩阵，误差相对运动的导数，左扰动模型求解
                    fx * pc[0] * pc[1] * inv_z2,   -fx - fx * pc[0] * pc[0] * inv_z2,    fx * pc[1] * inv_z,
                0,   -fy * inv_z,   fy * pc[1] * inv_z2,   
                    fy + fy * pc[1] * pc[1] * inv_z2,   -fy * pc[0] * pc[1] * inv_z2,    -fy * pc[0] * inv_z;

            H += J.transpose() * J; // 增量方程的Hessian矩阵。这里不含（或未知）噪声的方差
            b += -J.transpose() * e; // 增量方程的bias
        }

        Vector6d dx; 
        dx = H.ldlt().solve(b); // 求解增量线性方程。系数矩阵采LDLT分解。 

        if (isnan(dx[0])) { // 检测解是否存在
            cout << "result is nan! " << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost) { // 如果 cost 不降反增，则停止求解
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose; // 修正的相机位姿
        lastCost = cost;

        cout << "iteration" << iter << " cost = " << std::setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6) { // converage 收敛条件
            break;
        }
    }

    cout << "pose by g-n: \n" << pose.matrix() << endl;
}

/*-------3. BA求解，非线性问题通过g2o图优化求解，需派生节点和边的类-------*/
// g2o::BaseVertex 派生类，定义节点（优化变量）。
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> { // <> 第一个参数为优化变量维度，第二个为优化变量数据类型。6表示优化变量的局部参数空间的维度(独立元素？)，即李代数；SE3d为优化变量的数据类型，即李群。
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 宏定义，确保 new 分配的对象，在内存中按Eigen库对齐

    // virtual 虚函数，在派生类允许重写 override
    virtual void setToOriginImpl() override { 
        _estimate = Sophus::SE3d(); // 优化变量的初始值，为单位矩阵
    }

    // (左乘) left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen; // 更新量(李代数)
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate; // 优化变量的更新（通过在李群左乘exp()）
    }

    // 读盘和存盘（留空）
    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}
}; 

// g2o::BaseUnaryEdge 派生类，定义一元边（观测值）。
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> { // <> 第一个参数为观测值(同误差)维度，第二个参数为观测值(同误差)类型，第三个参数为连接的顶点类型。
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {} // 构造函数
    
    virtual void computeError() override { // 计算误差 （不是cost？）
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]); // 定义指向节点常量的指针（不能修改节点的值）。_vertices 是边基类的成员变量，用于存储连接到该边的顶点。
        Sophus::SE3d T = v->estimate(); // 返回所连接节点（优化变量）的估计值（李群）
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d); // 3D世界坐标 -> 像素坐标（未归一化）
        pos_pixel /= pos_pixel[2]; // 归一化坐标（预测值）
        _error = _measurement - pos_pixel.head<2>(); // _measurement观测值 为 g2o::BaseUnaryEdge 的保护变量，需在创建派生类时，从外部设置。
        // .head<2>() 头两个元素
    }

    virtual void linearizeOplus() override { // 雅可比矩阵。误差相对优化变量的雅可比矩阵。 
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]); // 同样是指向节点的值。
        Sophus::SE3d T = v->estimate(); // 同样是节点的估计值
        Eigen::Vector3d pos_cam = T * _pos3d; // 3D世界坐标 -> 相机坐标
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi << // 误差相对位姿的雅可比矩阵。通过左扰动模型计算。
            -fx / Z,   0,   fx * X / Z2,   fx * X * Y / Z2,   -fx - fx * X * X / Z2,   fx * Y / Z,
            0,   -fy / Z,   fy * Y / (Z * Z),   fy + fy * Y * Y / Z2, -fy * X * Y / Z2,   -fy * X / Z;
    }

    // 读盘和存盘
    virtual bool read(istream &in) override {} 
    virtual bool write(ostream &out) const override {}

    private: // 私有：通过构造函数传入后，不能外部访问，也不能派生类访问。
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

// BA+g2o求解
void bundleAdjustmentG2O(
        const VecVector3d &points_3d, const VecVector2d &points_2d,
        const Mat &K, Sophus::SE3d &pose) {
    // 初始：构建图优化，先设定g2o
    // g2o::BlockSolver 用于管理求解过程的优化变量(旋转向量+平移向量，6维)和中间量(地标点，3维)。通常与特定线性求解器使用。
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器
    // 梯度下降方法，GN、LM、DogLeg 可选，此处为GN
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()) );
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true); // 打开调试输出

    // vertex 节点（优化变量）
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose 由节点的类定义
    vertex_pose->setId(0); // 节点的标识符
    vertex_pose->setEstimate(Sophus::SE3d()); // 节点初始值。这里是按李群，不是李代数
    optimizer.addVertex(vertex_pose); // 将新的节点加入优化器

    // K 内参矩阵，转换为 Eigen 类型
    Eigen::Matrix3d K_eigen;
    K_eigen <<  K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
                K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
                K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges 边
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen); // 由边的类定义
        edge->setId(index); // 标识符
        edge->setVertex(0, vertex_pose); // 连接的顶点。参数：顶点的ID、指向顶点的指针
        edge->setMeasurement(p2d); // 输入观测值
        edge->setInformation(Eigen::Matrix2d::Identity()); // 信息矩阵（噪声协方差的逆），没有就按单位矩阵
        optimizer.addEdge(edge); // 将新的边加入优化器
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true); // 优化器输出详细信息：每一步迭代的损失值、梯度更新情况等
    optimizer.initializeOptimization(); // 优化器初始化
    optimizer.optimize(10); // 优化迭代的最大次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o = \n" << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate(); // 求得的节点值，即优化变量。estimate()访问节点里的_estimate值。
}