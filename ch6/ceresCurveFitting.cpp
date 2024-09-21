#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;


// 代价函数：y - exp(ax^2 + bx + c)
struct CURVE_FITTING_COST{
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}
    // 构造函数：和结构体名（或类名）相同的函数，调用结构体（或类）的同时即运行函数。
    // (double x, double y) 构造函数的参数列表，表示创建对象时需提供两个double型数据。
    // : _x(x), _y(y) {} 初始化列表，将传入的x和y分别赋值给成员变量_x和_y

    // 残差的计算
    template<typename T> // template<>模板函数。typename 指示T为类型参数（类似int、float），这里可以是任意类型，具体取决于使用。
    // bool 表示函数返回的为true或false的bool类型
    bool operator()( const T *const abc, T *residual) const// operator() 重载函数调用运算符()（不是重载函数），operator() 允许定义对象被"调用"时的行为。比如，obj(args)，args是传递给operator()的参数。
        // ()()：第一个()表示operator的重载；第二个()包含传递给operator()的参数列表。参数列表里一个为指向T类型的常量指针（用于模型参数），一个为指向T类型的指针（用于存储残差）。
        // const T *const abc 第一个const表示指针指向的数据是常量（不能通过指针修改指向的值，但能修改指针本身指向另一个地址），第二个const表示指针为常量（不能改变指针的值，但能改变指针指向的数据）。
        // 第二个()里的第一个参数，两个const表示，指针无法改变，指针指向的值也不能改变。
        // 使用指针abc指向T，为什么不是先定义一个T类型的变量：T是一个模板(template)参数，允许结构体与不同数据类型一起工作。
        // {前面的const是函数声明的部分，不是函数体定义{}里面的部分。表示该成员函数operator()是“常量成员函数”，保证在函数内部不会修改任何非静态成员变量的值。
        {
            residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // 残差块。结果存储在residual中。
            // T()用于对_x和_y的数据进行显示类型转换。（因为abc的定义类型即为T）
            return true; // 返回true表示计算成功
        }

    const double _x, _y; // _x和_y成员函数设定为常量化const：一旦在构造函数中初始化，就不能修改。（也可写在结构体开头）
};
// 结构体struct：用户自定的数据类型，可以像int、double那样使用，在函数的输入参数中需要写明struct。通常类型均为public，允许外部修改，且不含复杂的成员函数。
// 类class：自定的数据类型，与结构体相似。分为public、private、protected，通常封装采用类的private，以确保稳定性。

int main(int argc, char **argv){
    double ar = 1.0, br = 2.0, cr = 1.0; // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0; // 估计参数值
    int N = 100; // 用于拟合曲线的数据点数量
    double w_sigma = 1.0; // 噪声标准差 sigma 值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng; // OpenCV随机数产生器

    // 生成和存放N个数据点
    vector<double> x_data, y_data; 
    for (int i =0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    // 存放曲线的估计参数，作为初始值
    double abc[3] = {ae,be,ce};

    // 构建最小二乘问题
    ceres::Problem problem; // ceres求解问题的头
    for (int i = 0; i < N; i++){
        problem.AddResidualBlock( // 残差块
            // 自动求导<误差类型，输出维度，输入维度>。维数要与前面struct一致
            // 使用模板 CURVE_FITTING_COST 指定误差类型。1表示残差cost是一维的。3表示待求的参数向量abc是三维的。
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
                new CURVE_FITTING_COST(x_data[i], y_data[i]) // main函数上面，使用struct定义的残差块
            ),
            // new 用于在堆上动态分配相应类型的对象。一般不会直接new一个ceres::AutoDiffCostFunction给AddResidualBlock，可能导致内存泄漏（除非在其他地方显式地删除了）
            // 第一个<>里的 CURVE_FITTING_COST：一个结构体或类的类型。
            // 第二个()里的 CURVE_FITTING_COST：似乎不需要显式地创建？
            
            nullptr, // 核函数，这里为空（不使用）
            abc  // 待估计参数
        );
    }

    // 配置求解器
    ceres::Solver::Options options; // 具体可配置项参考文档
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // 增量方程的求解方式。DENSE_NORMAL_CHOLESKY 是 ceres 库的线性求解器之一。
    options.minimizer_progress_to_stdout = true; // 每次迭代后向cout输出当前的迭代信息。

    ceres::Solver::Summary summary; // 优化信息：完成优化过程后返回的结构体，包含了优化过程的详细信息，如迭代次数、最终cost、收敛状态、梯度评估次数等。
    
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); 
    
    ceres::Solve(options, &problem, &summary); // 开始优化
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    for (auto a:abc) cout << a << " ";
    cout << endl;
    
    return 0;
}