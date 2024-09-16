#include <iostream>
#include <fstream> // 提供对文件的输入和输出操作
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>  // 提供灵活的字符串格式化功能
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType; // 存取位姿的变量，用于生成轨迹
typedef Eigen::Matrix<double, 6, 1> Vector6d; // 六维向量类型

void showPointCloud( // 函数声明，在 Pangolin 中绘制点云
    const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv){
    vector<cv::Mat> colorImgs, depthImgs; // 定义彩色图和深度图变量
    TrajectoryType poses; // 定义相机的位姿序列变量

    // 读取pose.txt
    ifstream fin("./rgbd/pose.txt"); 
    // 检查pose.txt文件是否存在
    if (!fin) { // 若不存在，fin为错误状态(并不是fin包含一个bool值)；若存在，则fin读取文件内容，后续可用>>完成读取
        cerr << "请在有pose.txt的目录下运行此程序" << endl;
        return 1;
    }
    // 若存在，读取相应彩色图和深度图，以及pose.txt里的位姿信息
    for (int i = 0; i < 5; i++) { // pose.txt 包含了5个位姿信息（实际上对应5张彩色图和深度图）
        boost::format fmt("./rgbd/%s/%d.%s"); // boost::format 用于格式化字符串。
        // fmt 被定义为一个格式化模板。%s和%d为占位符，分别表示字符串和整数
        // 比如，下面调用的 fmt % "color" % (i + 1) % "png" 中，%s 被替换为 "color"，%d 被替换为 (i+1) 的值，以此实现图像文件路径的动态构造。
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str())); 
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1)); 
        // 若不设置第二个参数，按默认设置读取：RGB图像读取为BGR三通道，灰度图读取为单通道。
        // 第二个参数设置为-1,会以包括alpha通道的原始格式读取图像。对于pgm格式(深度图)，可以确保深度值不会被错误解释为颜色值。

        double data[7] = {0}; // 初始化声明：包含7个元素的 double 数组，并且所有元素初始化为0.
        // 注意：初始声明时可以采用 double data[7]，但后续不能使用 data[7] 操作(索引从0开始，data[7] 为第8个元素)。

        for (auto &d:data) // 以引用方式访问数组中的每个元素。
        // 使用引用可以避免不必要的复制，并允许在循环体内修改数组元素的值。
            fin >> d; 

        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2])); 
        poses.push_back(pose);
    }

    // 计算点云并拼接
    // 相机内参 
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0; // 通常为：深度图像素值转换为实际物理距离的比例因子
    // 定义点云
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000); // .reverse(size_t n) 请求容器预留足够的空间（并不改变容器大小，只是预留空间）

    for (int i = 0; i < 5; i++) {
        cout << "转换图像中: " << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                // .ptr<数据类型>(行)[列] 按设定的数据类型返回相应像素。
                // 深度图通常用16位整数(unsigned short)来存储深度值

                if (d == 0) continue; // 为0表示没有测量到

                Eigen::Vector3d point;
                point[2] = double(d) / depthScale; // 获取实际物理深度距离
                point[0] = (u - cx) * point[2] / fx; // (已校正畸变)像素坐标 -> 归一化坐标 -> *深度 相机坐标系
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point; // 相机坐标 -> 世界坐标

                Vector6d p;
                p.head<3>() = pointWorld; // .head<3>为前3个元素（索引从0开始）
                // cv::Mat 类提供了几种方法来访问不同颜色通道的像素值，如.data、.step、.channels()
                // .data 是指向图像数据首字母的指针。访问图像数据较复杂，需要自行处理图像的行填充和颜色通道的顺序。
                p[5] = color.data[v * color.step + u * color.channels()];   // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                // .step 返回图像中一行数据的字节数。（若图像行之间有填充字节，上述公式可能不正确）
                // .channels() 返回图像的颜色通道数。对于灰度图像是1,对于RGB图像是3.
                pointcloud.push_back(p);
            }
    }

    // 显示点云
    cout << "点云共有" << pointcloud.size() << "个点." << endl;
    showPointCloud(pointcloud);
    return 0;
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {
    // 判断文件是否读取成功
    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    // 初始化Pangolin，设定窗口大小
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 设定 s_cam 对象，定义相机的投影矩阵和模型视图矩阵。
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // 设置 d_cam 对象，与 s_cam 关联，并设置其在窗口中的位置和大小
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    // 标准 OpenGL 渲染循环，直到用户请求退出（例如，关闭窗口）。
    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}