#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groundtruth_file = "./example/groundtruth.txt"; // 引入真实轨迹
string estimated_file = "./example/estimated.txt"; // 引入估计轨迹

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType; 
// 重定义一个向量类型名TrajectoryType，后面用作函数的返回量类型

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

TrajectoryType ReadTrajectory(const string &path);
// 函数声明（定义在main函数后面），输入参数为路径指针

int main(int argc, char **argv) {
    // 按ReadTrajectory函数定义，这里读取文件路径，并返回元素为SE(3)的向量类型
    TrajectoryType groundtruth = ReadTrajectory(groundtruth_file); 
    TrajectoryType estimated = ReadTrajectory(estimated_file);

    // assert()函数，检查条件是否为真，若为假则中止并输出错误信息（通常用于测试）
    assert(!groundtruth.empty() && !estimated.empty());
    assert(groundtruth.size() == estimated.size());

    // 计算se(3)的均方根误差rmse，用于计算绝对轨迹误差
    double rmse = 0;
    for (size_t i = 0; i < estimated.size(); i++){
        // size_t为c++标准库中定义的无符号整型，.size()返回一个size_t类型的值
        
        Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
        double error = (p2.inverse() * p1).log().norm();
        rmse += error * error;
    }
    rmse = rmse / double(estimated.size());
    rmse = sqrt(rmse);
    cout << " RMSE = " << rmse << endl;

    DrawTrajectory(groundtruth, estimated);
}


// 读取轨迹文件里的轨迹数据（上一章只有一个文件，就没额外写函数，这里需要读取两个.txt文件）
TrajectoryType ReadTrajectory(const string &path) {
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << "trajectory " << path << " not found." << endl;
    return trajectory;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
    trajectory.push_back(p1);
  }
  return trajectory;
}


// 轨迹绘制的函数
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // 初始化Pangolin
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768); // 创建窗口并设定窗口像素大小
  glEnable(GL_DEPTH_TEST); // 启用深度测试
  glEnable(GL_BLEND); // 启用混合（blending）
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam( // 定义相机的投影矩阵和模型视图矩阵
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay() // d_cam为与s_cam关联的视图对象
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f) // 定义视图在窗口中的位置和大小
      // .SetBOunds()的参数分别表示d_cams视图的：水平方向的起始、结束位置，垂直方向的起始位置，垂直的高度，宽高比？
      // pangolin::Attach::Pix(175)表示：相对参考点偏离175像素
      .SetHandler(new pangolin::Handler3D(s_cam)); //// 绑定一个 Handler3D，允许通过鼠标和键盘与相机交互


  while (pangolin::ShouldQuit() == false) { // 标准的 OpenGL 渲染循环，直到用户请求退出（关闭窗口等）。
    // pangolin::ShouldQuit() 请求退出（按Esc、关闭窗口等）时，返回true
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清除屏幕和深度缓冲区

    d_cam.Activate(s_cam); // 激活与d_cam关联的s_cam的渲染状态
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 设置OpenGL的清除颜色为纯白（即屏幕颜色）

    glLineWidth(2); // 设置OpenGL线条绘制的线宽

    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES); // 指定接下来的顶点用于绘制线段 
      auto p1 = gt[i], p2 = gt[i + 1]; // auto为自动类型推导
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]); // .translation()用于提取平移向量
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd(); 
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame(); // 完成所有OpenGL调用后的标准结束语（交换缓冲区并检查事件）
    usleep(5000);   // sleep 5 ms
  }

}