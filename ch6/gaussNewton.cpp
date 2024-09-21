#include <iostream>
#include <chrono> // 用于计时
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv){
    double ar = 1.0, br = 2.0, cr = 1.0; // 曲线的真实参数
    double ae = 2.0, be = -1.0, ce = 5.0; // 曲线的估计参数（未知参数初始值）
    int N = 100; // 用于拟合的数据点数量
    double w_sigma = 1.0; // 高斯分布噪声的标准差sigma值
    double inv_sigma = 1.0 / w_sigma; // sigma的逆？
    cv::RNG rng; // OpenCV随机数产生器。类型可以指定为整数或浮点数，但不直接生成矩阵或数组。

    vector<double> x_data, y_data; // x_data存放输入的x数据；y_data存放带高斯噪声的y数据
    for (int i = 0; i < N; i++){
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma)); // 生成标准高斯噪声 rng.gaussian(w_sigma * w_sigma)
    }

    // Gauss-Newton 迭代
    int iterations = 100; // 迭代次数
    double cost = 0, lastCost = 0; // 本次迭代的cost和上次迭代的cost（cost function所得值）

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++){

        Matrix3d H = Matrix3d::Zero(); // Hessian矩阵
        Vector3d b = Vector3d::Zero(); // bias?
        cost = 0; // cost function的值

        // 计算增量方程的系数矩阵H和非齐次的值b，计算最小二乘的cost function
        for (int i = 0; i < N; i++){
            double xi = x_data[i], yi = y_data[i]; // 读取第i个数据
            double error = yi - exp(ae * xi * xi + be * xi + ce); // 观测值（带噪声的真值）和初始估计值的误差
            Vector3d J; // 雅可比矩阵

            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce); // 误差e对待拟合参数a的偏导
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce); // 误差e对待拟合参数b的偏导
            J[2] = -exp(ae * xi * xi + be * xi + ce); // 误差e对待拟合参数c的偏导

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error; // cost function的值
        }

        // 求解线性方程 Hx = b
        Vector3d dx = H.ldlt().solve(b); // 求解线性方程。LDLT分解，近似对称正定的矩阵分解为下三角矩阵L、对角矩阵D及其转置LT。
        if (isnan(dx[0])) { // isnan() 检查是否为非数字(NaN)
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
            break;
        }

        // 迭代修正曲线参数
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;

        cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
         "\t\testimated params: " << ae << "," << be << "," << ce << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    // chrono::duration 提供可用于运算的时间间隔；chrono::duration_cast 提供间隔数据的单位和类型的转换
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}