#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

string file_1 = "./LK1.png"; 
string file_2 = "./LK2.png";

// Optical flow tracker and interface
class OpticalFlowTracker {// 用于跟踪两帧图像间的特征点的光流
    public:
    OpticalFlowTracker( const Mat &img1_,   const Mat &img2_,
        const vector<KeyPoint> &kp1_,   vector<KeyPoint> &kp2_,   
        vector<bool> &success_,    bool inverse_ = true,   bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_) {} // 构造函数

    void calculateOpticalFlow( const Range &range); // 计算指定范围内特征点光流的成员函数，在main后面定义

    private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true; // 指示启用 反向(Inverse)光流法
    bool has_initial = false; // 指示启用 初始估计
};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(  
    const Mat &img1,   const Mat &img2,
    const vector<KeyPoint> &kp1,   vector<KeyPoint> &kp2,
    vector<bool> &success,   bool inverse = false,   bool has_initial_guess = false);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default 
 * the imgae pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2 
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,   const Mat &img2,   
    const vector<KeyPoint> &kp1,   vector<KeyPoint> &kp2,
    vector<bool> &success,   bool inverse = false);

/**
 * 从灰度图像获取指定位置(x,y)的像素值的函数，使用双线性插值计算非整数坐标处的像素值。
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */
inline float GetPixelValue(const cv::Mat &img, float x, float y){
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x); // floor() 向下取整。此处用减法取小数部分。
    float yy = y - floor(y); 
    int x_a1 = std::min(img.cols - 1, int(x) + 1); // 相邻像素的索引（同时考虑是否超出边界）
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) // 双线性插值，利用四个相邻像素的值，和它们与(x,y)点的相对距离，计算(x,y)点的像素值
        + xx * (1 - yy) * img.at<uchar>(y, x_a1) 
        + (1 - xx) * yy * img.at<uchar>(y_a1, x)
        + xx * yy * img.at<uchar>(y_a1, x_a1);
}


int main(int argc, char **argv) {
    Mat img1 = imread(file_1, 0); // Imgaes,  灰度图 CV_8UC1
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here. GFTT算法提取关键点。
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints.
    // 500 最多返回的关键点数量， 0,01 质量水平参数（接受的关键点的最低质量）， 20 块大小参数（计算角点检测中的协方差矩阵，值越大、对噪声越不敏感，但会增加计算量） 
    detector->detect(img1, kp1);

    // Track key points in the second image
    /*-------------------2.单层光流 single level LK --------------------*/
    cout << endl << " single level LK: "  << endl;
    vector<KeyPoint> kp2_single; // 声明 vector 类型的变量时，默认初始化为空向量。
    vector<bool> success_single;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by single level LK: " << time_used.count() << " s" << endl;

    /*-----------------------------------------------------------------*/

    /*--------------------3.多层光流 multi-level LK ----------------------*/
    cout << endl << " multi-level LK: "  << endl;
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << " s" << endl;
    /*-------------------------------------------------------------------*/

    /*-----------------1.OpenCV的光流 cv::calcOpticalFlowPyrLK 函数---------------------*/
    cout << endl << " OpenCV LK flow: "  << endl;
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    // img1 前一帧图像， img2 后一帧图像， pt1 前一帧的特征点， pt2 后一帧的特征点， status 每个特征点正确匹配时依次为1的输出数组， error 每个特征点的误差估计的输出数组。
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by OpenCV: " << time_used.count() << " s" << endl;
    /*--------------------------------------------------------------------------------*/

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by OpenCV", img2_CV);
    cv::waitKey(0);

    return 0;
}

/*-----------------2.单层光流函数-----------------*/
void OpticalFlowSingleLevel(
        const Mat &img1,   const Mat &img2, 
        const vector<KeyPoint> &kp1,   vector<KeyPoint> &kp2,
        vector<bool> &success,   bool inverse,   bool has_initial) { 
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial); // tracker 不是函数，而是这里定义的 OpticalFlowTracker 类的实例。参数列表传入构造函数。
    parallel_for_(Range(0, kp1.size()),   // cv::parallel_for 并行for循环，Intel tbb库实现。
        std::bind( &OpticalFlowTracker::calculateOpticalFlow,  &tracker,  placeholders::_1 )  );
        // 并行调用 OpticalFlowTracker::calculateOpticalFlow 计算指定范围内特征点的光流。
        // Range(0, kp1.size()) 创建了一个从 0 到 kp1.size()-1 的整数范围的 Range 对象，用于指定 parallel_for_ 函数应该并行处理的迭代范围。
        // std::bind 一个函数模板，将函数及其参数绑定。比如这里将 pticalFlowTracker::calculateOpticalFlow 成员函数与 tracker 实例绑定。
        // placeholders::_1 为一个占位符，表示 parallel_for_ 传递给 calculateOpticalFlow 的当前迭代索引。根据上下文为从 0 到 kp1.size()-1 的整数。
        // calculateOpticalFlow 在 OpticalFlowTracker 类中声明为接受一个 Range 类型的参数，但这里通过 std::bind 绕过，利用 parallel_for_ 的迭代机制为每个迭代调用。
}

void OpticalFlowTracker::calculateOpticalFlow( const Range &range) { // 类的成员默认可直接访问类的成员变量（包括私有和共有）。来自类外部的数据需要作为参数传递（有时出于规范，将所用的成员变量写在参数列表）
    // 光流类 OpticalFlowTracker 的成员函数 calculateOpticalFlow 的定义，用于计算指定范围内特征点的光流
    int half_patch_size = 4; // 基于某一窗口内像素运动相同的假设，定义每个特征点周围图像块的半边长（单位为像素），用于计算光流
    int iterations = 10; // 优化过程的最大迭代次数

    for (size_t i = range.start; i < range.end; i++) { // range 表示什么意义呢？
        auto kp = kp1[i]; 
        double dx = 0, dy = 0; // dx,dy need to be estimated. dx 和 dy 表示特征点在 x 和 y 方向的位移。
        if (has_initial) { // 启用初始估计，即先查看是否有dx和dy
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        } // 启用初始估计时，应先检查 kp2 是否为空，比如 kp2.size() 检查

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded. 后面存入success数组，表示该特征点光流计算成功。

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero(); // Hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero(); // bias
        Eigen::Vector2d J; // Jacobian
        for (int iter = 0; iter < iterations; iter++) { // 非线性迭代优化
            if (inverse == false) { // 设定是否启用 反向(Inverse)光流法，保持梯度（即雅可比）不变（Hessian也不变），只计算残差
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }
            else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // compute cost and Jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++){
                for (int y = - half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1,   kp.pt.x + x,   kp.pt.y + y)  
                                - GetPixelValue(img2,   kp.pt.x + x + dx,   kp.pt.y + y + dy) ; 
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d( // 不启用反向光流时，利用相邻像素差值计算梯度（雅可比）
                            0.5 * (GetPixelValue(img2,   kp.pt.x + dx + x + 1,   kp.pt.y + dy + y) - GetPixelValue(img2,   kp.pt.x + dx + x - 1,   kp.pt.y + dy + y)) ,
                            0.5 * (GetPixelValue(img2,   kp.pt.x + dx + x,   kp.pt.y + dy + y + 1) - GetPixelValue(img2,   kp.pt.x + dx + x,   kp.pt.y + dy + y - 1))    );
                    }
                    else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // Note this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d( // 若启用反向光流，计算梯度时不考虑dx和dy，实质上对同一像素为固定值。
                            0.5 * (GetPixelValue(img1,   kp.pt.x + x + 1,   kp.pt.y + y) - GetPixelValue(img1,   kp.pt.x + x - 1,   kp.pt.y + y)) ,
                            0.5 * (GetPixelValue(img1,   kp.pt.x + x,   kp.pt.y + y + 1) - GetPixelValue(img1,   kp.pt.x + x,   kp.pt.y + y - 1))    ); 
                    }

                    // compute H, b and set cost;
                    b += -error * J; // 依照高斯牛顿法的增量方程计算
                    cost += error * error; // 通过 min cost 求解 dx 和 dy
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }
            }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) { // 增量方程求解失败
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan " << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) { // 下降至极小值
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) { // 求解结果收敛
                // converge
                break;
            }
        }

        success[i] = succ; // 存储该特征点是否成功求解(匹配？)光流

        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy); // dx和dy即为非线性优化求解得到的光流偏移量（根据灰度不变假设）
    }
}
/*-----------------------------------------------------------------*/

/*-----------------------3.多层光流法（通过多次调用单层光流函数）-------------------------------*/
void OpticalFlowMultiLevel(
        const Mat &img1,    const Mat &img2,
        const vector<KeyPoint> &kp1,   vector<KeyPoint> &kp2,
        vector<bool> &success,   bool inverse) {
    // parameters
    int pyramids = 4; // 4层金字塔
    double pyramid_scale = 0.5; // 每层缩放倍率为0.5
    double scales[] = {1.0, 0.5, 0.25, 0.125}; // 每层依次的缩放倍率

    // create pyramids. 创建金字塔
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2; // imgae pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else{
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1],   img1_pyr,  // 解释下cv::resize函数及其参数
                    cv::Size(pyr1[i - 1].cols * pyramid_scale,   pyr1[i - 1].rows * pyramid_scale)); // 解释下cv::Size函数及其参数
            cv::resize(pyr2[i - 1],   img2_pyr,
                    cv::Size(pyr2[i - 1].cols * pyramid_scale,   pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "build pyramid time: " << time_used.count() << endl;

    // coarse-to-fine LK tracking in pyramids. 由粗至精过程
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) {
        auto kp_top = kp ;
        kp_top.pt *= scales[pyramids - 1];  // 最顶层缩放后的像素。直接像素坐标乘倍数，即跟随图像尺寸比例调整位置。
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);  // 这里有没有kp2_pyr关系不大，后面 OpticalFlowSingleLevel 函数里会更新输出。如果启用了初始估计，赋值能确保不会报错。
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true); // 调用单层光流函数，计算结果作为下一层初始值
        t2 = chrono::steady_clock::now();
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;

        if (level > 0) {
            for (auto &kp: kp1_pyr)     kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)     kp.pt /= pyramid_scale;
        }
    }

    for (auto &kp: kp2_pyr)     kp2.push_back(kp); // &kp:kp2_pyr 表示for循环里的kp是引用的kp2_pyr(实际为kp2_pyr)的元素？最后放入kp2？
}
/*-----------------------------------------------------------------------------*/