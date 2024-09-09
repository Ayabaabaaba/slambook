#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv){
    // 创建旋转矩阵：沿Z轴转90度。AngleAxisd()用于创建旋转向量
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
    
    // 构建双精度SO(3)：可以从旋转矩阵，也可以从四元数，二者等价
    Sophus::SO3d SO3_R(R);
    Quaterniond q(R);
    Sophus::SO3d SO3_q(q);
    cout << "SO(3) from matrix:\n" << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << endl;
    cout << "they are equal" << endl;

    // 李群 SO(3) -> 李代数 so(3)：对数映射 .log()
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl ;
    // hat：向量 -> 矩阵 （ so(3) 为 李代数 -> 反对称矩阵 ）
    cout << "so3 hat =\n" << Sophus::SO3d::hat(so3) << endl;
    // vee：矩阵 -> 向量
    cout << "so3 hat vee = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;
    
    // 旋转矩阵SO(3)引入微小扰动 ΔR·R 
    Vector3d update_so3(1e-4, 0, 0); // 定义更新量
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << " SO3 updated = \n" << SO3_updated.matrix() << endl;

    cout << "*********" << endl;

    // 旋转 + 平移 定义 变换矩阵SE(3)
    Vector3d t(1, 0, 0); // 沿x轴平移1
    Sophus::SE3d SE3_Rt(R, t); // 从 旋转矩阵R、平移向量t 构造SE(3)
    Sophus::SE3d SE3_qt(q, t); // 从 四元数q、平移向量t 构造SE(3)
    cout << "SE3 from R,t = \n" << SE3_Rt.matrix() << endl;
    cout << "SE3 from q,t = \n" << SE3_qt.matrix() << endl;
    
    // 李群 SE(3) -> 李代数 se(3)：对数映射 .log()
    typedef Eigen::Matrix<double, 6, 1> Vector6d; // se(3)为六维向量，typedef以方便定义
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl; // 在Sophus中，se(3)平移在前，旋转在后
    // hat：向量 -> 矩阵 
    cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
    // vee：矩阵 -> 向量
    cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

    // 变换矩阵SE(3)引入微小扰动 ΔT·T
    Vector6d update_se3; // 定义更新量
    update_se3.setZero();
    update_se3(0, 0) = 1e-4; //(0,0)表示矩阵第一个元素。
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

    return 0;
}