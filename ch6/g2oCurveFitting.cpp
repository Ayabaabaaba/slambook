#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

// 类成员的类型
// public：对外可见，对内可见
// private：对外不可见，对内可见。（未声明时默认为private）
// protected：对外不可见，对内可见，且对派生类可见。

// 类的继承：一个类可以从现有类中派生，而不必重新定义新类
// class 新派生类名标识符 : [继承方式] 旧基类名标识符
// 继承方式 public （共有型派生）：基类的 public、protected 在派生类仍为 public、protected（可访问），基类的 private 在派生类不可访问。
// 继承方式 private （私有型派生）：基类的 public、protected 在派生类为 private（类体内可访问，类外不可访问，次派生类不可访问），基类的 private 在派生类不可访问。
// 继承方式 protected （保护型派生）：基类的public、protected在派生类均为protected（类体内可访问，类外不可访问，次派生类可访问），基类的 private 在派生类不可访问。

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>{
    // g2o::BaseVertex<> 为g2o库中用于表示顶点的基类模板。第一个参数为优化变量维度，第二个为优化变量数据类型。
    // 本例中，优化变量为三维的Eigen向量

    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // 一个宏定义，确保通过 new 分配的对象，在内存中按Eigen库的要求对齐。
    // 这个宏定义是在Eigen库中定义的，故这里没有#define等（Eigen库中有）

    // virtual：在基类中声明虚函数，在派生类中允许重写 override
    // 通常派生类中不需要再写 virtual，写上可增加可读性，影响不大。
    // 若派生类中函数与基类虚函数同名，但参数列表或返回类型不同，则不是重写，而是隐藏了基类中的虚函数。
    // 重置
    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0; // _estimate 变量在 g2o::BaseVertex 基类中定义，并且是 protected
        // _estimate 的类型由模板参数 Eigen::Vector3d 决定
        // << 0, 0, 0 是Eigen库中向量的初始化方式，将前三个元素分别初始化为0
    }

    // 更新
    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update); // 相当于 Eigen::Vector3d(update[0], update[1], update[2])
        // update 在 oplusImpl 方法中通常来自优化迭代过程中计算出的增量（更新量）
    }

    // 存盘和读盘：将数据保存到文件，从文件读取数据。
    // read 和 write 函数用于顶点的序列化和反序列化。
    // 此处 read 和 write 的实现为空，意味着没有实现具体的逻辑，可能不需要将顶点数据保存到文件(或稍后实现)。

    // 读盘和存盘（留空）
    virtual bool read(istream &in) {} // istream 是C++标准库中的输入流类，&in 是对 istream 类型对象的引用。
    virtual bool write(ostream &out) const {} // ostream 是输出流类，&out是对 ostream 类型对象的引用。
    // const 表明在 write 函数不会修改类的成员变量（除了被声明为 mutable 的）。
    // write 函数一般只用于输出数据，不改变对象状态，因此声明 const 是一个好习惯。
    // read 函数本身就要读取数据，若声明为const则无法成功读取。
};


// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
    // g2o::BaseUnaryEdge<> 为g2o库中表示一元边(只连接一个顶点)的基类模板
    // <> 第一个参数为观测值的维度，第二个参数为观测值的类型，第三个参数为该边所连接的顶点的类型

    public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {} 
    // 构造函数。
    // BaseUnaryEdge()为对基类构造函数的调用。

    // 计算曲线模型误差（Cost Function）
    virtual void computeError() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        // CurveFittingVertex 作为类名，也可用于声明或定义变量、指针。
        // static_cast<> 为C++的类型转换操作符。与动态类型转换(dynamic_cast)不同，static_cast主要用于基本数据类型或类的继承体的转换，并且转换前不检查类型（若转换不安全，编译时不报错，运行时可能报错）
        // <>内是想要转换到的目标类型。const CurveFittingVertex * 即指向 CurveFittingVertex 常量的指针，即不能通过这个指针修改 CurveFittingVertex 对象。
        // ()内是被转换的对象。_vertices 是g2o::BaseUnaryEdge基类的成员变量，用于存储连接到该边的顶点（对于一元边，_vertices数组大小为1）。
        // 经const定义，似乎CurveFittingVertex的值就是不可修改的？
        

        const Eigen::Vector3d abc = v->estimate();
        // v是指向CurveFittingVertex的指针。estimate()是CurveFittingVertex类的一个成员函数，用于返回当前顶点的估计值。
        // abc = v->estimate() 从指针v指向的类中，调用函数estimate()，并将返回值赋给abc

        _error(0,0) =_measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x +abc(2, 0));
        // Eigen::Vector3d 本身是通过Matrix定义的，在索引时需要通过行列索引
        // 在C++标准容器（如std::vector），使用 abc[2] 索引是可行的，但在Eigen的Matrix中，需要使用()索引行列。

    }

    // 计算雅可比矩阵
    virtual void linearizeOplus() override {
        const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
        _jacobianOplusXi[0] = -_x * _x * y;
        _jacobianOplusXi[1] = -_x * y;
        _jacobianOplusXi[2] = -y;
    }

    // 读盘和存盘（留空）
    virtual bool read(istream &in) {}
    virtual bool write(ostream &out) const{}

    public:
    double _x; // x值（y值为_measurement）
};


int main(int argc, char **argv) {
    double ar = 1.0, br = 2.0, cr = 1.0; // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
    int N = 100; // 数据点
    double w_sigma = 1.0; // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma; 
    cv::RNG rng;  // OpenCV随机数产生器

    // 观测数据集
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // typedef 创新类型别名

    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType; // 优化变量维度为3,误差值维度为1
    // g2o::BlockSolver 是 g2o 中的一个关键类，用于管理求解过程中需要的各种矩阵和向量。通常与特定线性求解器（如 g2o::LinearSolverDense 或 g2o::LinearSolverCholesky）一起使用
    // g2o::BlockSolverTraits<3,1> 是一个模板参数，用于指定优化变量的维度（这里是3）和单个误差项的维度（这里是1）
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // g2o::LinearSolverDense<> 一个线性求解器，用于解决g2o的线性系统。需要一个矩阵类型作为模板参数，这里为 BlockSolverType::PoseMatrixType
    // BlockSolverType::PoseMatrixType 为 BlockSolverType 内部定义的类型，表示优化过程中使用的矩阵类型。

    // new 用于手动动态分配内存（需要使用delete释放内存，可能导致内存泄漏或重复释放等问题）
    // 函数模板：C++中泛型编程的一种实现方式，允许程序员编写与类型无关的函数。建立一个通用的函数，其参数类型和返回类型不具体指定，而是用一个虚拟的类型（称为模板参数）来代表。
    // 函数模板的声明通常为： template<typename T> 或 template<class T>

    // 梯度下降法，可以从GN、LM、DogLeg中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton( // 这里是选择了GN的梯度下降法。这里似乎可以不用 new
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())
        // C++14标准后，就存在std::make_unique，而新版的 g2o 则移除了make_unique 函数
        // std::make_unique 是一个函数模板，用于创建一个 std::unique_ptr 的实例。
        // BlockSolverType 和 LinearSolverType 应该是两种类类型。
        // () 里的参数，应该是作为构造函数（第一个实际上是 BlockSolverType 的构造函数）的传入参数
    );
    g2o::SparseOptimizer optimizer; // 图模型
    // g2o::SparseOptimizer 是 g2o 库中的图优化模型类，用于存储和管理图中的所有顶点和边，提供了设置求解其、添加顶点和边、执行优化等功能。
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true); // 打开调试输出


    // 往图中增加顶点
    CurveFittingVertex  *v = new CurveFittingVertex();
    // CurveFittingVertex 首先作为类型声明 v 的类型（一个指向 CurveFittingVertex 对象的指针）。接着 CurveFittingVertex() 作为构造函数名，创建这个类型的实例。
    // new 将 CurveFittingVertex() 创建的实例分配在堆(heap)上，而指针 v 存储了该实例的内存地址。
    v->setEstimate(Eigen::Vector3d(ae, be, ce)); // 这里是调用了指针 v 指向的 CurveFittingVertex 类的成员函数吧？setEstimate和setId函数是什么作用，里面输入的参数是什么作用？
    v->setId(0);
    // v-> 调用指针 v 指向的 CurveFittingVertex 类的成员函数。
    // setEstimate() 设置顶点（优化变量）的估计值。
    // setId() 给顶点设置一个唯一的标识符(ID)。在图中，每个顶点都需要一个唯一的ID区分和引用（通常接受整数参数）。
    optimizer.addVertex(v); 
    // optimizer 是一个 g2o::SparseOptimize 类型的对象（用于表示图优化模型的类）。
    // addVertex 是 g2o::SparseOptimizer 类的一个成员函数，用于将新的顶点添加到优化器中。addVertex 函数接受一个指向顶点对象的指针作为参数。

    // 往图中增加边
    for (int i = 0; i < N; i++){
        CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0, v);  // 设置连接的顶点。两个参数为：顶点的ID索引，指向顶点的指针
        edge->setMeasurement(y_data[i]); // 观测数值
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // 信息矩阵：协方差矩阵之逆（噪声方差的逆）
        // 前面设置误差项时，就用Eigen库的Matrix设置顶点类型，这里是统一类型。
        optimizer.addEdge(edge);
    }

    // 执行优化
    cout << "start optimization " << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10); // 10表示优化迭代的最大次数
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds." << endl;

    // 输出优化值
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;

    return 0;
}