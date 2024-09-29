#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

// 函数声明
void find_feature_matches( // 寻找匹配的特征点对（去掉误匹配）
    const Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches
);

void pose_estimation_2d2d( // 2D2D图像的运动估计
    std::vector<KeyPoint> keypoints_1,
    std::vector<KeyPoint> keypoints_2,
    std::vector<DMatch> matches,
    Mat &R, Mat &t
);

Point2d pixel2cam(const Point2d &p, const Mat &K); // 像素坐标 -> 相机归一化坐标
// Point2d 一个类型声明，表示一个二维点（有2个double型数据，表示x和y坐标）
// const Point2d &p 表明函数体内不会修改p指向的对象。Mat 为 OpenCV 库中用于存储图像或矩阵的类。K 通常表示相机的内参矩阵，包含相机焦距和光学中心等参数。


int main(int argc, char **argv){
    if (argc != 3){ // argc的数量，为1个程序名+2个argv(链接到图像)
        cout << "ussage: pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }

    //读取图像
    Mat img_1 = imread(argv[1], IMREAD_COLOR); // 按彩色模式读取图像
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!"); // 字符串始终作为非空值，只是出现假报错时，会输出整行代码（包括字符串信息）

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches); // 
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 估计两张图像间的运动
    Mat R,t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // 验证 E=t^R 
    // Mat_<double>(3, 3) 为 cv::Mat 类的一个模板特化，允许创建矩阵时，指定类型和大小（这里为3*3的double型）
    Mat t_x = (Mat_<double>(3, 3) << 
                0,  -t.at<double>(2, 0),  t.at<double>(1, 0), 
                t.at<double>(2, 0),  0,  -t.at<double>(0, 0),
                -t.at<double>(1, 0),  t.at<double>(0, 0),  0);

    cout << "t^ R = " << endl << t_x * R << endl;

    // 验证对极约束
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 内参矩阵：(fx, 0, cx, 0, fy, cy, 0, 0, 1)
    for (DMatch m: matches) { 
        // matches 是一个 std::vector<DMatch> 类型的容器，存储了匹配的特征点对信息。
        // DMatch m: matches 将 matches 的每个元素都遍历一遍，依次赋值给 变量m

        // pixel2cam 为自定义函数，用于 二维像素坐标 -> 二维归一化坐标
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);

        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl; // 按照对极约束，不同特征点对计算后理论上都是0
    }
    return 0;
}


/*------------------*/
// 使用OpenCV库，检测和匹配ORB特征点对的函数封装，包含误匹配筛选
void find_feature_matches(const Mat &img_1, const Mat &img_2,
                        std::vector<KeyPoint> &keypoints_1,
                        std::vector<KeyPoint> &keypoints_2,
                        std::vector<DMatch> &matches){
    // 初始化
    Mat descriptors_1, descriptors_2;
    
    // 提取ORB特征（OpenCV3函数）
    // Ptr<> 是 OpenCV 中用于智能指针的模板类，<>为指针指向的。ORB::create() 初始化为ORB特征检测器（工厂方法）
    Ptr<FeatureDetector> detector = ORB::create(); 
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // 使用汉明距离的暴力匹配器

    // 第一步：检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1); // 根据前面定义，变量detector 为指向ORB特征点检测器的指针。
    detector->detect(img_2, keypoints_2);
    // 第二步：根据角点位置计算BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    // 第三步：匹配两幅图的BRIEF描述子，使用Hamming距离
    vector<DMatch> match; // DMatch 为包含两个描述子间匹配信息的结构体。
    matcher->match(descriptors_1, descriptors_2, match); // 根据前面的定义，matcher 设置的为使用汉明距离的暴力匹配器。

    // 第四步：匹配点对筛选
    double min_dist = 10000, max_dist = 0;
    // 找出所有匹配间的最小距离和最大距离，即最相似和最不相似的两组点的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n ", max_dist);
    printf("-- MIn dist : %f \n ", min_dist);
    // 误匹配筛选：筛掉描述子间距离大于两倍最小距离的。最小距离下限按经验设置为30
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)){
            matches.push_back(match[i]);
        }
    }

}


// 像素坐标 -> 归一化坐标 (用于验证对极约束)
Point2d pixel2cam(const Point2d &p, const Mat &K){
    // Point2d 是 OpenCV 定义的用于表示二维点的类，包含两个double型变量x和y
    // const Point2d &p 表示不能通过p修改其指向的Point2d型变量，但可以访问。
    // 也可用 const Point2d *p ，此时 p.x 和 p.y 改为 p->x 和 p->y 。比起指针*，更常用引用&，可以避免空指针的问题。

    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),  // (p.x - cx) / fx
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)   // (p.y - cy) / fy
    );
}


// 匹配的特征点对 -> 基础矩阵、本质矩阵、单应矩阵 -> 相机运动R、t
void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                        std::vector<KeyPoint> keypoints_2,
                        std::vector<DMatch> matches,
                        Mat &R, Mat &t) {
    // 相机内参，TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); 
    // 内参矩阵逐行写为：(fx, 0, cx, 0, fy, cy, 0, 0, 1)
    // fx=αf  fy=βf  f为焦距，α、β分别为像素坐标相对于成像平面坐标的缩放倍数。(fx、fy也可理解为焦距的x、y分量？)
    // cx、cy 为像素坐标原点相对于成像坐标原点的偏移。（也可理解为光心在像素平面的偏移坐标？）

    // 匹配点转换为vector<Point2f>形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // 计算基础矩阵 F
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    // 计算本质矩阵 E
    Point2d principal_point(325.1, 249.7); // 相机光心，TUM dataset标定值？
    double focal_length = 521.0; // 相机焦距，TUM dataset标定值
    Mat essential_matrix;
    // essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    essential_matrix = findEssentialMat(points1, points2, K);  // 
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    // 计算单应矩阵 H（本例中不是平面，单应矩阵意义不大）
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    // 分解本质矩阵（OpenCV3函数）
    // recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point); 
    recoverPose(essential_matrix, points1, points2, K, R, t); // 
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}   
