#include <iostream>
#include <vector>
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv){
    Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
    q1.normalize(); // 单位化，包括实部和虚部
    q2.normalize();
    Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3); // 定义机器人位置，即平移向量(相对世界系)
    Vector3d p1(0.5, 0, 0.2); // 定义观测点坐标

    Isometry3d T1w(q1), T2w(q2); // 四元数(仅表示旋转) -> 欧氏变换矩阵
    T1w.pretranslate(t1); // 在变换矩阵中添加平移向量
    T2w.pretranslate(t2);

    Vector3d p2 = T2w * T1w.inverse() * p1; // 观测点坐标经两次变换矩阵变换
    cout << endl << p2.transpose() << endl;
    return 0;
}
