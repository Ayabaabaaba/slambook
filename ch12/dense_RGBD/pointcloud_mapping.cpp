#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

int main(int argc, char **argv) {
    vector<cv::Mat> colorImgs, depthImgs;  // 彩色图和深度图
    vector<Eigen::Isometry3d> poses;  // 相机位姿

    ifstream fin("./dense_RGBD/data/pose.txt");  // 读取位姿数据 
    if (!fin) {
        cerr << "cannot find pose file" << endl;
        return 1;
    }

    for (int i = 0; i < 5; i++) {
        boost::format fmt("./dense_RGBD/data/%s/%d.%s");   // 图像文件格式
        colorImgs.push_back( cv::imread( (fmt % "color" % (i + 1) % "png").str() ) );
        depthImgs.push_back( cv::imread( (fmt % "depth" % (i + 1) % "png").str(), -1 )); // -1 读取原始图像

        double data[7] = {0}; // 此处为第一个元素显式初始化为0。其他元素隐式初始化为0（不显式初始化也会自动为0）
        for (int i = 0; i < 7; i++ ) {
            fin >> data[i]; // 依次读取位姿数据
        }
        Eigen::Quaterniond q(data[6], data[3], data[4], data[5]); 
        Eigen::Isometry3d T(q);
        T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2])); // 设置变换矩阵的平移部分
        poses.push_back(T);
    }

    // 计算点云拼接
    // 相机内参
    double cx = 319.5;
    double cy = 239.5;
    double fx = 481.2;
    double fy = -480.0;
    double depthScale = 5000.0; // 深度尺度缩放因子

    cout << "正在将图像转换为点云... " << endl;

    // 定义点云使用的格式（此处为XYZRGB）
    typedef pcl::PointXYZRGB PointT;  // 点的类型
    typedef pcl::PointCloud<PointT> PointCloud;  // 点云的类型

    // 新建一个点云
    PointCloud::Ptr pointCloud(new PointCloud); // PointCloud::Ptr 是PCL使用的一个智能指针。此处指向一个PointCloud对象，并使用 new 动态分配内存（建议结尾显式调用 delete）
    for (int i = 0; i < 5; i++) {
        PointCloud::Ptr current(new PointCloud);
        cout << "转换图像中：" << i + 1 << endl; 
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Eigen::Isometry3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) { // 遍历所有像素
                unsigned int d = depth.ptr<unsigned short>(v)[u]; // 读取像素对应的深度值
                if (d == 0)  continue;  // 为0表示没有测到 
                Eigen::Vector3d point;  // 相机坐标 = 相机归一化坐标 * 深度缩放因子
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point; // 相机坐标 -> 世界坐标

                PointT p; // 设置点云的坐标和颜色
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[v * color.step + u * color.channels()]; // BGR中，每个通道的像素值平铺为一维数据。
                p.g = color.data[v * color.step + u * color.channels() + 1]; // .step 表示 color 中从一个颜色数据块移动到下一颜色数据块的补偿（每个颜色通道组之间的间隔）
                p.r = color.data[v * color.step + u * color.channels() + 2]; // .channels() 返回通道数。比如BGR为3
                current->points.push_back(p);
            }
        // depth filter and statical removal. 对当前点云应用统计滤波，去除噪声点，并将滤波后的点云添加到总点云中。
        PointCloud::Ptr tmp(new PointCloud);
        pcl::StatisticalOutlierRemoval<PointT> statistical_filter; // pcl::StatisticalOutlierRemoval<> 用于执行统计离开群点移除算法（统计滤波器），<>为点云中点的类型。
        statistical_filter.setMeanK(50); // 设置用于计算平均距离的邻域点数
        statistical_filter.setStddevMulThresh(1.0); // 设置标准差阈值倍数
        statistical_filter.setInputCloud(current); // 设置输入点云
        statistical_filter.filter(*tmp); // 执行滤波，结果存储在tmp变量中
        (*pointCloud) += *tmp; // 将滤波后的点云添加到总点云
    }

    pointCloud->is_dense = false; // 设置点云为非稠密(因为可能存在无效点)。is_dense 为 pcl::PointCloud 类的一个成员变量，用于指示点云是否被认为是“稠密”的。
    cout << "点云共有" << pointCloud->size() << "个点。" << endl;

    // voxel filter. 体素滤波器
    pcl::VoxelGrid<PointT> voxel_filter;
    double resolution = 0.03; // 体素滤波器的分辨率（即每个体素格子的长度）
    voxel_filter.setLeafSize(resolution, resolution, resolution);  // resolution. 在x, y, z三个方向上设置相同的体素大小 
    PointCloud::Ptr tmp(new PointCloud); // 创建一个新的点云指针tmp，用于存储滤波后的点云数据  
    voxel_filter.setInputCloud(pointCloud); // 设置体素滤波器的输入点云  
    voxel_filter.filter(*tmp); // 执行滤波操作，将结果存储在tmp指向的点云对象中 
    tmp->swap(*pointCloud); // 使用swap方法，将tmp中的点云数据，交换到原来的pointCloud指针指向的对象中。// 这样，原来的pointCloud就包含了滤波后的点云数据，而tmp则变为空

    cout << "滤波之后，点云共有" << pointCloud->size() << "个点。" << endl;

    pcl::io::savePCDFileBinary("./dense_RGBD/build/map.pcd", *pointCloud); // 保存点云到.pcd文件，可用pcl_viewer工具查看
    return 0;
}