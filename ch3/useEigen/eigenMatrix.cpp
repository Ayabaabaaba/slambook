#include <iostream>
using namespace std;

#include<ctime>

#include<eigen3/Eigen/Core> // Eigen核心
#include<eigen3/Eigen/Dense> // 稠密矩阵的代数运算
using namespace Eigen;

#define MATRIX_SIZE 50 // 宏定义变量 MATRIX_SIZE 为 50

/*Eigen基本使用*/

int main(int argc, char **argv){
    //Eigen::Matrix<>用于定义矩阵或向量，前三个参数分别为数据类型、行、列
    Matrix<float,2,3> matrix_23; //声明一个2*3的float矩阵

    //Eigen通过typedef提供许多内置类型，底层仍是Eigen::Matrix
    Vector3d v_3d; // Vector3d 实质是 Eigen::Matrix<double,3,1>
    Matrix<float,3,1> vd_3d; //同样是3*1向量，但为float型

    Matrix3d matrix_33 = Matrix3d::Zero(); // Matrix3d 同 Eigen::Matrix<double,3,3>。
    // 这里初始化为零

    Matrix<double, Dynamic, Dynamic> matrix_dynamic; // 动态大小矩阵，用于不确定矩阵大小时
    MatrixXd matrix_x; // 同前一行一致

    // Eigen矩阵的输入输出
    matrix_23 << 1,2,3,4,5,6;
    cout << "matrix 2x3 are respectively: \n" << matrix_23 <<endl;

    //矩阵中单个元素的访问 (i, j)
    cout << "matrix 2x3:" <<endl;
    for(int i =0; i<2; i++ ){
        for(int j=0; j<3; j++) cout << matrix_23(i,j) <<"\t";
        cout << endl;
    }

    // 矩阵乘法。Eigen的乘法需要同类型矩阵，或者矩阵具有显式转换
    v_3d << 3,2,1; // double型
    vd_3d << 4,5,6; // float型
    Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d; // 不同类型需有显式转换
    cout << "[1,2,3;4,5,6]*[3;2;1]=" << result.transpose() << endl;
    Matrix<float, 2, 1> result2 = matrix_23 * vd_3d; // 同类型可以直接相乘（矩阵维度合适时）
    cout << "[1,2,3;4,5,6]*[4;5;6]=" << result2.transpose() << endl;

    // 矩阵的其他运算
    matrix_33 = Matrix3d::Random();      // 随机数矩阵
    cout << "random matrix: \n" << matrix_33 << endl;
    cout << "transpose: \n" << matrix_33.transpose() << endl;      // 转置
    cout << "sum: " << matrix_33.sum() << endl;            // 各元素和
    cout << "trace: " << matrix_33.trace() << endl;          // 迹
    cout << "times 10: \n" << 10 * matrix_33 << endl;               // 数乘
    cout << "inverse: \n" << matrix_33.inverse() << endl;        // 逆
    cout << "det: " << matrix_33.determinant() << endl;    // 行列式

    // 特征值分解
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;
    // 实对称矩阵可以确保对角化

    // 解线性方程组：matrix_NN * x = v_Nd
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE); // 定义matrix_NN并赋随机值
    matrix_NN = matrix_NN * matrix_NN.transpose(); // 确保半正定，即对任意x有 x^T A x >= 0 恒成立，在最小二乘问题中能确保得到全局最优解
    Matrix<double, MATRIX_SIZE, 1 > v_Nd = MatrixXd::Random(MATRIX_SIZE,1); // 定义v_Nd并赋随机值
    // 直接求逆的解法及其所耗时长
    clock_t time_stt = clock(); // 获取当前时间以便计时
    Matrix<double , MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd; // 通过求逆解方程组
    cout<<"time of normal inverse is " << 1000 *(clock() - time_stt) / (double) CLOCKS_PER_SEC <<"ms" << endl; // 输出所用时间
    cout << "x=" << x.transpose() << endl; // 输出求逆的求解结果
    // 矩阵分解（如QR分解）可能会更快？
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of QR decomposition is " << 1000 *(clock() - time_stt) / (double) CLOCKS_PER_SEC <<"ms" << endl;
    cout << "x = " << x.transpose() << endl;
    // 正定矩阵还能用cholesky分解来求解
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is " << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    return 0;
}