#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "./imageBasics/distorted.png";

int main(int argc, char **argv) {
    // 这里使用公式推导畸变，而不是使用内置函数 cv::Undistort()，有助于理解
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05; // 畸变参数
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375; // 内参

    cv::Mat image = cv::imread(image_file, 0);   // 读取图像。此处为灰度图，CV_8UC1
    
    int rows = image.rows, cols = image.cols;
    cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1);   // 定义一个变量，用于表示去畸变后的图

    // 计算去畸变后的图像内容，并赋值给 image_undistort
    for (int v = 0; v < rows; v++){ // 数组行对应图像高
        for (int u = 0; u < cols; u++){ // 数组列对应图像宽

            // 计算点(u,v)对应到畸变图像中的坐标(u_distorted, v_distorted)
            // 从像素平面（未校正畸变）反推到归一化平面（未校正畸变）
            double x = (u - cx) / fx, y = (v - cy) / fy; 
            double r = sqrt(x * x + y * y);
            // 对归一化坐标进行畸变校正
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            // 从校正后的归一化坐标到像素坐标（已校正）
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
                image_undistort.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            } else {
                image_undistort.at<uchar>(v, u) = 0;
            }
            // .at<uchar>(y,x) 是OpenCV中访问图像像素值的方法，uchar表示无符号字符类型(unsigned char，用于表示图像像素)
        }
    }

    // 显示畸变前后的图像
    cv::imshow("distorted", image);
    cv::imshow("undistorted", image_undistort);
    cv::waitKey();
    return 0;
}