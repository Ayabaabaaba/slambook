#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
    const  Mat &img_1, const Mat &img_2,
    std::vector<KeyPoint> &keypoints_1,
    std::vector<KeyPoint> &keypoints_2,
    std::vector<DMatch> &matches);

void pose_estimation_2d2d(
    const std::vector<KeyPoint> &keypoints_1,
    const std::vector<KeyPoint> &keypoints_2,
    const std::vector<DMatch> &matches,
    Mat &R, Mat &t);

void triangulation( // 三角化的函数声明
    const vector<KeyPoint> &keypoints_1,
    const vector<KeyPoint> &keypoints_2,
    const std::vector<DMatch> &matches,
    const Mat &R, const Mat &t,
    vector<Point3d> &points);

// 用于绘图
inline cv::Scalar get_color(float depth) { // inline 内联函数，在调用点内联展开函数体。主要用于较小但频繁调用的函数体，减少函数调用的开销。
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range)); 
    // 根据深度值，分别计算R、G、B通道的颜色。这里是从蓝色(低depth)到红色(高depth)渐变的颜色。
}

// 像素坐标 -> 归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);


int main(int argc, char **argv){
    if (argc != 3) {
        cout << "ussage: triangulation img1 img2" << endl;
        return 1;
    }

    // 读取图像
    Mat img_1 = imread(argv[1], IMREAD_COLOR);
    Mat img_2 = imread(argv[2], IMREAD_COLOR);

    // 匹配特征点
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 估计运动
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    // 三角化
    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points); // 自定的三角化函数，里面用 cv::triangulatePoints 函数求解三角化后的点坐标

    // 验证三角化点与特征点的重投影关系？
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 相机的内参矩阵
    Mat img1_plot = img_1.clone(); // .clone() 复制图像矩阵
    Mat img2_plot = img_2.clone();
    for (int i = 0; i < matches.size(); i++){
        // 第一张图
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K); // 将第一张图中匹配的像素坐标，转换为相机归一化坐标
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2); // 在图像的特征点上画圆圈。get_color为inline函数

        // 第二张图
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z ) + t; // 三角化后的点坐标为第一张图的坐标系，第一坐标系经R、t才到第二坐标系。
        float depth2 = pt2_trans.at<double>(2, 0); //（0, 0）为第一个元素，(2, 0) 为第三个元素
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2); 
    }

    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

    return 0;
}

// 匹配两张图的特征点
void find_feature_matches(const Mat &img_1, const Mat &img_2, std::vector<KeyPoint> &keypoints_1, std::vector<KeyPoint> &keypoints_2, std::vector<DMatch> &matches) {
    // 初始化
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 第一步：检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    // 第二步：根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    // 第三步：匹配BRIEF描述子，使用 Hamming 距离的暴力匹配
    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    // 第四步：筛选误匹配
    double min_dist = 1000, max_dist = 0;
    for (int i = 0; i < descriptors_1.rows; i++){
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }

}

// 2d2d估计运动
void pose_estimation_2d2d( const std::vector<KeyPoint> &keypoints_1, const std::vector<KeyPoint> &keypoints_2, const std::vector<DMatch> &matches, Mat &R, Mat &t) {
    // 相机内参
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // 匹配的特征点单独提取至 vector<Point2f>
    vector<Point2f> points1;
    vector<Point2f> points2;
    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt); // .pt 为 Point2f 类型的变量，表示关键点的坐标(x, y)
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    // 计算本质矩阵
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, K);

    // 分解本质矩阵 -> 旋转 + 平移
    recoverPose(essential_matrix, points1, points2, K, R, t);
}

// 三角化函数？
void triangulation(
    const vector<KeyPoint> &keypoint_1, const vector<KeyPoint> &keypoint_2,
    const std::vector<DMatch> &matches, const Mat &R, const Mat &t, 
    vector<Point3d> &points) {

        Mat T1 = (Mat_<float>(3,4) << 1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 1, 0); // 变换矩阵（无旋转平移）
        Mat T2 = (Mat_<float>(3,4) << 
            R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
        
        Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
        vector<Point2f> pts_1, pts_2;
        for (DMatch m:matches) { // 像素坐标 -> 归一化坐标
            pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
            pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K)); 
        }
        cout << pts_1 << endl;
        cout << pts_2 << endl;

        Mat pts_4d;
        cv::triangulatePoints(T1, T2, pts_2, pts_1, pts_4d); 
        // triangulatePoints() 用于三角化的函数。
        // 输入参数分别为：第一个相机的变换矩阵，第二个相机的变换矩阵，第一个图像中匹配的特征点集，第二个图像中匹配的特征点集，输出存储三角化后三维点坐标的矩阵（每个三维点以四维齐次坐标存放于一列）

        // 转换成非齐次坐标?
        for (int i = 0; i < pts_4d.cols; i++) {
            Mat x = pts_4d.col(i);
            x /= x.at<float>(3,0); // 归一化（三角化后得到相机系？）
            Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
            );
            points.push_back(p);
        }
}   

// 像素坐标 -> 归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}
