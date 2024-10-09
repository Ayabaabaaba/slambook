#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

// 相比于类，结构体的成员均是public的。结构体也需要通过定义对象作为实例操作。
struct PoseAndIntrinsics{ // 9维相机参数，姿态(位姿)和内参
    PoseAndIntrinsics() {} // 构造函数

    // set from given data address
    explicit PoseAndIntrinsics( double *data_addr ) { // 构造函数参数列表不同，就构成了重载。此处为读取9维相机参数
    // explicit 要求显式调用，以防在只有一个实参时被隐式调用进行类型转换。比如， PoseAndIntrinsics obj = data; 错误， PoseAndIntrinsics obj(data); 正确
        rotation = SO3d::exp( Vector3d(data_addr[0], data_addr[1], data_addr[2] ) ); // data_addr 9维相机参数，包括3维旋转向量、3维平移向量、1维焦距、2维畸变系数
        translation = Vector3d( data_addr[3], data_addr[4], data_addr[5] );
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    // 估计值放入内存
    void set_to(double *data_addr) { // 将9维相机参数存入指定数组
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i)  data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i)  data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focal = 0; // 焦距
    double k1 = 0, k2 = 0; // 畸变系数
};

// 顶点（位姿和相机内参）：9维，3维so3+三维t+f+k1+k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> { // 表示相机的顶点。通过派生类隐式创建PoseAndIntrinsics结构的实例对象（顶点内可使用结构体函数）。
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // 宏定义，确保通过 new 分配的对象，在内存中按Eigen库的要求对齐。

    VertexPoseAndIntrinsics() {} // 构造函数

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics(); // 优化变量初始值，为空？
    }

    virtual void oplusImpl(const double *update) override { // 更新优化变量
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

    // 根据估计值投影一个点（路标点估计值的预测像素坐标）
    Vector2d project(const Vector3d &point) { // project() 在该顶点的实例中可使用
        Vector3d pc = _estimate.rotation * point + _estimate.translation; // 世界系 -> 相机系
        pc = -pc / pc[2]; // 归一化
        double r2 = pc.squaredNorm(); 
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);// -> 去畸变 -> 像素坐标 
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};

// 顶点（路标点3维坐标）
class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};

// 边（观测）
class EdgeProjection : public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
// g2o::BaseBinaryEdge 为连接两个顶点的边。2为观测值维度（此处为像素坐标）；Vector2d 为观测值类型（此处为Eigen库的类型，便于Eigen库计算）；VertexPoseAndIntrinsics 为连接的第一个顶点的类型；VertexPoint 为连接的第二个顶点的类型
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override { // 计算观测和预测的误差（可能作为cost function的一部分）
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0]; // 第一个顶点的指针
        auto v1 = (VertexPoint *) _vertices[1]; // 第二个顶点的指针
        auto proj = v0->project(v1->estimate()); // 路标点估计值的投影像素坐标（预测值）
        _error = proj - _measurement;
    }

    // 此处没有雅可比矩阵的解析形式，自动求导

    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const {}
};

void SolveBA(BALProblem &bal_problem); // 使用g2o求解BA问题的函数声明

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "ussage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("./build/initial_g2o.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("./build/final_g2o.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size(); // 路标点坐标的维度3
    const int camera_block_size = bal_problem.camera_block_size(); // 相机参数的维度10或9
    double *points = bal_problem.mutable_points(); // BAL数据中，路标点3D坐标的起始位置
    double *cameras = bal_problem.mutable_cameras(); // BAL数据中，相机参数的起始位置

    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType; // g2o::BlockSolver 管理求解过程的优化变量(相机参数)和中间量(路标点)。
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType; // 指定线性求解器为 CSparse 库（一个稀疏矩阵求解器库）
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg( // 梯度下降方法为LM，也可换乘GN、DogLeg
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())     );
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true); // 打开调试输出

    // build g2o problem
    const double *observations = bal_problem.observations(); // BAL数据中，观测数据起始位置
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics; // 管理指向顶点的指针的容器
    vector<VertexPoint *> vertex_points;
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics(); // 定义指向顶点的指针
        double *camera = cameras + camera_block_size * i;
        v->setId(i); // 节点ID
        v->setEstimate(PoseAndIntrinsics(camera)); // 节点初始值
        optimizer.addVertex(v); // 添加节点到优化器
        vertex_pose_intrinsics.push_back(v);  // 管理节点指针
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras()); // 即使类不同，节点的ID也不能重复
        v->setEstimate(Vector3d(point[0], point[1], point[2])); 
        // g2o在BA中需要手动设置待Marg的顶点
        v->setMarginalized(true); // 标记一个顶点为被边缘化（marginalized）。用于系数结构的矩阵，在计算时整合到边缘化先验。
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]); // 连接顶点。顶点ID、指向顶点的指针。（有一个管理指针的容器就方便很多）
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1])); // 输入观测值
        edge->setInformation(Matrix2d::Identity()); // 信息矩阵（噪声协方差的逆），没有就按单位矩阵
        edge->setRobustKernel(new g2o::RobustKernelHuber()); // 鲁棒核函数，设置为Huber核
        optimizer.addEdge(edge); // 加入边到优化求解器
    }

    optimizer.initializeOptimization(); // 优化器初始化
    optimizer.optimize(40); // 设置优化迭代的最大次数

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate(); // 依次读取每个相机参数的估计值
        estimate.set_to(camera); // 这里相当于重置cameras指向数组的数据
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k)   point[k] = vertex->estimate()[k]; // 依次读取每个路标点坐标的估计值，并重置到points指向数组的数据
    }
}
