#include <iostream>
#include <cmath>
using namespace std;

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
using namespace Eigen;

int main(int argc, char **argv){ //argc表示命令中字符串的个数，用空格分隔，直接运行可执行程序时为1。argv数组包含用户在命令行传递的字符串命令，通常第一个字符串为程序名
    // 定义旋转矩阵 & 旋转向量
    Matrix3d rotation_matrix = Matrix3d::Identity(); // 使用 Matrix3d 或 Matrix3f 定义3x3旋转矩阵：赋值为单位矩阵
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1)); // 使用 AngleAxisd 定义旋转向量和旋转角(因重载了运算符，可以当作矩阵运算)：沿 z 轴旋转45度。
    // 重载：同名不同参数，便于用同一个名字表示函数或符号作用

    cout.precision(3); // 输出小数点后三位
    cout << "rotation matrix =\n" << rotation_vector.matrix() << endl;   // .matrix()：将旋转向量转换成矩阵
    rotation_matrix = rotation_vector.toRotationMatrix();
    cout << rotation_matrix << endl; // 对旋转向量使用.matrix()和.toRotationMatrix()效果似乎一致
    // cout << rotation_matrix.matrix() << endl; // 结果与前面一致

    // 旋转向量或旋转矩阵进行坐标变换
    Vector3d v(1, 0, 0); // 定义待旋转的向量
    Vector3d v_rotated = rotation_vector * v; // 使用旋转向量进行旋转
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;
    v_rotated = rotation_matrix * v; // 使用旋转矩阵进行旋转
    cout << "(1,0,0) after rotataion (by matrix) = " << v_rotated.transpose() << endl; // 旋转向量 or 旋转矩阵 直接左乘的结果一致

    // .eulerAngles() // 旋转矩阵 -> 欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX顺序，即yaw-pitch-roll
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    // Eigen::Isometry // 定义欧氏变换矩阵，会多出一维
    Isometry3d T = Isometry3d::Identity(); // 命令为3d，实际为4*4矩阵
    T.rotate(rotation_vector);  // 将旋转向量rotation_vector代入为矩阵T的旋转矩阵部分
    T.pretranslate(Vector3d(1,3,4));  // 将矩阵T的评议向量设定为(1,3,4)向量
    cout << "Transofrm matrix = \n" << T.matrix() << endl;
    // 变换矩阵进行坐标变换
    Vector3d v_transformed = T * v; // 包含旋转和平移
    cout << "v transformed = " << v_transformed.transpose() << endl;

    // 仿射变换和射影变换，通过 Eigen::Affine3d 和 Eigen::Projective3d 定义

    // 四元数
    // Quaterniond // 定义四元数；旋转向量->四元数;旋转矩阵->四元数
    Quaterniond q = Quaterniond(rotation_vector); // 定义 + 旋转向量->四元数
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
    // .coeffs()：返回包含四个系数(3虚部+1实部)的向量。顺序为(x,y,z,w)，前三个为虚部，w为实部
    v_rotated = q * v; // 四元数旋转一个向量，重载的乘法，数学上为qvq^{-1}
    // cout << "v = " << v.transpose() << endl;
    cout << "(1,0,0) after rotation (by quaternion) = " << v_rotated.transpose() << endl;
    cout << "should be equal to " << (q * Quaterniond(0,1,0,0) * q.inverse()).coeffs().transpose() << endl; // 实际的数学运算过程
    // cout << Quaterniond(0,1,0,0).coeffs().transpose() << endl;
}
