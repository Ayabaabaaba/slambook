#include <opencv2/opencv.hpp>
#include <vector> 
#include <string>
#include <Eigen/Core> // 矩阵运算，线性代数
#include <pangolin/pangolin.h>  // 李群李代数
#include <unistd.h>

using namespace std;
using namespace Eigen;

// 读取左右视觉图像
string left_file = "./stereo/left.png";
string right_file = "./stereo/right.png";

// 函数声明：通过pangolin画图（定义在main后面）
void showPointCloud(
    const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud);

int main(int argc, char ** argv){
    
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157; // 内参
    double b = 0.573; // 基线：两相机光圈中心的距离

    // 读取图像
    cv::Mat left = cv::imread(left_file, 0); // 第二个参数'0'表示以灰度模式读取；若为1或省略，将以彩色模式读取（保留所有颜色通道）
    cv::Mat right = cv::imread(right_file, 0);
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32); 
    // cv::Ptr 智能指针类。在这里是创建一个指向 cv::StereoSGBM 对象的智能指针。
    // cv::StereoSGBM::create 用于创建StereoSGBM对象，一种用于计算立体图像对之间视差的算法。
    // 第一个参数'0'：指定预设的算法模式，0表示默认设置。
    // 后续参数为设定的视差、像素范围，按神奇参数设定即可。
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    // 调用 StereoSGBM 对象的 compute 方法，计算两个输入图像之间的视差图。
    // -> C++的成员访问运算符，用于通过指针访问类的成员。
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
    // 将 disparity_sgbm 中的视差图，转换为 CV_32F(浮点数类型，比默认CV_16S的16位有符号整数的精度更高)，并乘以缩放因子1.0/16.0f
    // .convertTo() 函数是 cv::Mat 类的一个成员函数，用于将矩阵转换为另一种数据类型，并可选地应用缩放和偏移。

    // 生成点云
    vector<Vector4d, Eigen::aligned_allocator<Vector4d>> pointcloud; // 定义存放点云的数组
    for (int v = 0; v < left.rows; v++){
        for (int u = 0; u < left.cols; u++){
            if (disparity.at<float>(v,u) <= 0.0 || disparity.at<float>(v,u) >= 96.0) continue; // 如果视差为负，或超出设定的最大视差(（sgbm的第二个参数），则跳过该像素

            Vector4d point(0, 0, 0, left.at<uchar>(v,u) / 255.0); // 定义4维向量 point ，前三维为 xyz，第四维为颜色

            // 双目模型计算 point 位置
            // 像素坐标 -> 归一化坐标
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            // 深度信息计算
            double depth = fx * b / (disparity.at<float>(v, u));
            // 归一化坐标 -> 相机坐标
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth; // 点云实际为三维坐标点组成

            pointcloud.push_back(point); // .push_back() 在vector数组的末尾添加元素。
        }
    }

    // 显示视差图
    cv::imshow("disparity", disparity / 96.0 );
    // disparity / 96.0 用于将视差归一化。96为 sgbm 中第二个参数最大视差对应的值。
    // cv::imshow 函数期望的输入图像为一个8位无符号整数(CV_8U)或浮点型(CV_32F)的图像，其像素值通常在[0, 255](对于 CV_8U)或[0,1](对于 CV_32F)
    cv::waitKey(0);
    // 使用自定义的函数，画出点云
    showPointCloud(pointcloud);
    return 0;
}

// 使用Pangolin库，绘制点云图
void showPointCloud( const vector<Vector4d, Eigen::aligned_allocator<Vector4d>> &pointcloud){
    if (pointcloud.empty()){
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    // 初始化Pangolin
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768); // 窗口设定
    glEnable(GL_DEPTH_TEST); // 启用深度和混合
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // s_cam 用于定义相机的投影矩阵和模型视图矩阵。
    pangolin::OpenGlRenderState s_cam( 
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // d_cam 是一个与 s_cam 关联的视图对象，这里设置了其在窗口中的位置和大小
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // 一个标准的 OpenGL 渲染循环，直到用户请求退出（例如，关闭窗口）。
    while (pangolin::ShouldQuit() == false) {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清除屏幕和深度缓冲区
        d_cam.Activate(s_cam); // 激活与d_cam关联的s_cam的渲染状态
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 设置OpenGL的清除颜色为纯白（即屏幕颜色）

        glPointSize(2); // 设置OpenGL绘制点的尺寸大小

        glBegin(GL_POINTS); // 指定接下来的顶点用于绘制点
        for (auto &p: pointcloud) { // 遍历 pointcloud 的每个元素，这里 pointcloud 是 std::vector<Vector4d, Eigen::aligned_allocator<Vector4d>> 类型
        // p[0]、p[1]、p[2]和p[3]分别访问了该向量的第一、二、三、四个元素
        // 按主函数中的定义，p[0]、p[1]、p[2]分别对应xyz坐标，p[3]对应归一化的灰度颜色？
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();

        pangolin::FinishFrame(); // 完成所有OpenGL调用后，交换缓冲区并检查事件
        usleep(5000);   // sleep 5 ms
    }
    return;
}