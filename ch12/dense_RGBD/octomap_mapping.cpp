#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h>    // for octomap 

#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;  // 彩色图和深度图
    vector<Eigen::Isometry3d> poses;  // 相机位姿

    ifstream fin("./dense_RGBD/data/pose.txt"); // 读取位姿文件
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++ ) { // 一共是5组图
        boost::format fmt("./dense_RGBD/data/%s/%d.%s"); // 图像文件格式
        colorImgs.push_back( cv::imread( (fmt % "color" % (i + 1) % "png").str() ) ); // 读取彩色图
        depthImgs.push_back( cv::imread( (fmt % "depth" % (i + 1) % "png").str(), -1 ) ); // 读取深度图。-1读取原始图像

        double data[7] = {0};
        for (int i = 0; i < 7; i++ ) {
            fin >> data[i]; // 7个位姿数据
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(T);
    }

    // 计算点云并拼接
    // 相机内参
    double cx = 319.5;
    double cy = 239.5;
    double fx = -480.0;
    double fy = -480.0;
    double depthScale = 5000.0;

    cout << "正在将图像转换为 Octomap ... " << endl;

    // octomap tree
    octomap::OcTree tree(0.01);  // 参数为分辨率
    for (int i = 0; i < 5; i++) {
        cout << "转换图像中：" << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];

        octomap::Pointcloud cloud; // the point cloud in octomap. 定义一个OctoMap的点云对象
        for (int v = 0; v < color.rows; v++) 
            for (int u = 0; u < color.cols; u++) { // 遍历图像每个像素
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
                if (d == 0)  continue;   // 为0表示没有测到

                
                Eigen::Vector3d point; // 相机系坐标
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point; // 相机系 -> 世界系
                
                cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]); // 将世界坐标系的点放入点云
            }

        // 将点云存入八叉树地图，给定相机位置作为投射线的起点
        tree.insertPointCloud(cloud, octomap::point3d( T(0, 3), T(1, 3), T(2, 3) )  );
        // .insertPointCloud() 为octomap八叉树的成员函数，用于将点云插入到八叉树中。
    }

    // 更新中间节点的占据信息，并写入磁盘
    tree.updateInnerOccupancy(); // 更新八叉树地图中节点的占据信息。
    cout << "saving octomap ... " << endl;
    tree.writeBinary("./dense_RGBD/build/octomap.bt"); 
    return 0;
}