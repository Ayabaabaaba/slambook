#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Core>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;
using Sophus::SE3d;
using Sophus::SO3d;

typedef Matrix<double, 6, 6> Matrix6d;
typedef Matrix<double, 6, 1> Vector6d; // 李代数顶点

// 误差右雅可比J_r^{-1}的近似 
Matrix6d JRInv(const SE3d &e) {
    Matrix6d J;
    J.block(0, 0, 3, 3) = SO3d::hat(e.so3().log()); 
    // .block(0, 0, 3, 3) 从第0行第0列开始的3*3子块
    // e.so3().log() 从SE3d对象 e 中提取出 SO3d 旋转部分，再对数映射到李代数。SO3d::hat() 将三维向量映射到3*3反对称矩阵
    J.block(0, 3, 3, 3) = SO3d::hat(e.translation()); // e.translation() 从e提取出平移向量
    J.block(3, 0, 3, 3) = Matrix3d::Zero(3, 3);
    J.block(3, 3, 3, 3) = SO3d::hat(e.so3().log());

    J = J * 0.5 + Matrix6d::Identity(); 
    // J = Matrix6d::Identity(); // 近似为单位阵
    return J;
}

// 顶点派生类，相机位姿李代数。
class VertexSE3LieAlgebra : public g2o::BaseVertex<6, SE3d> { // 此处定义6维优化变量。即便用四元数，在Sophus库中也能自动转换。
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual bool read(istream &is) override { // 读取节点的位姿参数
        double data[7];
        for (int i = 0; i < 7; i++)   is >> data[i];
        setEstimate( SE3d( 
            Quaterniond(data[6], data[3], data[4], data[5]), // 数据中旋转用四元数表示。SE3d变量的旋转部分能存储四元数，在Sophus库中可自行转换，所以开头定义的为6维。
            Vector3d(data[0], data[1], data[2])
        ));
    }

    virtual bool write(ostream &os) const override { 
        // os 是一个输出流对象（比如std::ofstream、std::cout），通过输出流运算符<<写入数据（cout也用<<输出）
        os << id() << " "; // id()是顶点派生类的成员函数，返回顶点的ID
        Quaterniond q = _estimate.unit_quaternion(); // _estimate在这里是一个Sophus::SE3d对象，通过.unit_quaternion()提取单位四元数（表示纯旋转）。
        os << _estimate.translation().transpose() << " "; // 先写入平移向量。.transpose()并不必需，写上可以保持代码的清晰性。
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << endl; // .coeffs() 是 Eigen::Quaterniond 类的成员函数，返回一个包含四元数系数的数组（需注意四元数实部虚部的顺序）。
        return true;
    }

    virtual void setToOriginImpl() override { // 优化变量初始值
        _estimate = SE3d();
    }

    virtual void oplusImpl(const double *update) override { // 更新优化变量
        Vector6d upd;
        upd << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3d::exp(upd) * _estimate;
    }
};

// 边派生类，两个相机李代数节点间的相对变化
class EdgeSE3LieAlgebra : public g2o::BaseBinaryEdge<6, SE3d, VertexSE3LieAlgebra, VertexSE3LieAlgebra> { // 观测维度，观测变量类型，两个连接顶点的类型
    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool read(istream &is) override { // 读取观测值
        double data[7];
        for (int i = 0; i < 7; i++)    is >> data[i];
        Quaterniond q(data[6], data[3], data[4], data[5]);
        q.normalize(); // 单位化四元数
        setMeasurement(  SE3d( q, Vector3d(data[0], data[1], data[2]) )  ); // 设置（边）观测值
        for (int i = 0 ; i < information().rows() && is.good(); i++) // .information() 返回边的信息矩阵（协方差矩阵的逆），信息矩阵维度与观测变量类型有关。 is.good() 检查文件流正确读取。
            for (int j = i; j < information().cols() && is.good(); j++) {
                is >> information()(i, j);
                if (i != j)    information()(j, i) = information()(i, j); // 协方差矩阵是对称的，其逆（信息矩阵）也是对称的。
            }
        return true;
    }

    virtual bool write(ostream &os) const override { // 写入观测值
        VertexSE3LieAlgebra *v1 = static_cast<VertexSE3LieAlgebra *> (_vertices[0]); // 起始顶点
        VertexSE3LieAlgebra *v2 = static_cast<VertexSE3LieAlgebra *> (_vertices[1]); // 终点顶点
        os << v1->id() << " " << v2->id() << " ";
        SE3d m = _measurement;
        Eigen::Quaterniond q = m.unit_quaternion();
        os << m.translation().transpose() << " ";
        os << q.coeffs()[0] << " " << q.coeffs()[1] << " " << q.coeffs()[2] << " " << q.coeffs()[3] << " ";

        // information matrix
        for (int i = 0; i < information().rows(); i++)
            for (int j = i; j < information().cols(); j++) {
                os << information()(i, j) << " ";
            }
        os << endl;
        return true;
    }

    virtual void computeError() override { // 误差计算
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        _error = (_measurement.inverse() * v1.inverse() * v2).log();
    }

    virtual void linearizeOplus() override { // 误差的雅可比计算
        SE3d v1 = (static_cast<VertexSE3LieAlgebra *> (_vertices[0]))->estimate();
        SE3d v2 = (static_cast<VertexSE3LieAlgebra *> (_vertices[1]))->estimate();
        Matrix6d J = JRInv(SE3d::exp(_error));
        // 可以尝试替换 J_r 的近似
        _jacobianOplusXi = -J * v2.inverse().Adj();
        _jacobianOplusXj = J * v2.inverse().Adj();
    }
};


int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: pose_graph_g2o_SE3_lie sphere.g2o" << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    vector<VertexSE3LieAlgebra *> vertices;
    vector<EdgeSE3LieAlgebra *> edges;
    while (!fin.eof()) {
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") {
            // 顶点
            VertexSE3LieAlgebra *v = new VertexSE3LieAlgebra();
            int index = 0;
            fin >> index;
            v->setId(index);
            v->read(fin);
            optimizer.addVertex(v);
            vertexCnt++;
            vertices.push_back(v);
            if (index == 0)   v->setFixed(true);
        } else if (name == "EDGE_SE3:QUAT") {
            // SE3-SE3 边
            EdgeSE3LieAlgebra *e = new EdgeSE3LieAlgebra();
            int idx1, idx2; // 关联的两个顶点
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++);
            e->setVertex(0, optimizer.vertices()[idx1]);
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin);
            optimizer.addEdge(e);
            edges.push_back(e);
        }
        if (!fin.good()) break;
    }

    cout << "read total " << vertexCnt << " vertices," << edgeCnt << " edges." << endl;

    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30); // 最大迭代次数

    cout << "saving optimization results ..." << endl;

    // 自定义顶点没有向g2o注册，无法直接输出.g2o文件
    // 这里自行实现伪装成 SE3 顶点和边（VERTEX_SE3:QUAT和EDGE_SE3:QUAT），让 g2o_viewer 可以认出
    ofstream fout("./build/result_lie.g2o"); // 创建一个输出文件流 fout。 ()操作符用于指定输出文件的路径和名称。
    for (VertexSE3LieAlgebra *v:vertices) {
        fout << "VERTEX_SE3:QUAT ";
        v->write(fout);
    }
    for (EdgeSE3LieAlgebra *e:edges) {
        fout << "EDGE_SE3:QUAT ";
        e->write(fout);
    }
    fout.close();
    return 0;
}
