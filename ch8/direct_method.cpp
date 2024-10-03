#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

using namespace std;

typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// Camera intrinsics 相机内参
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573; // 双目相机的基线
// paths
string left_file = "./left.png"; // 直接定义文件路径
string disparity_file = "./disparity.png";
boost::format fmt_others("./%06d.png");  // 格式化字符串，%06d 为一个格式化占位符，表示一个至少6位的十进制整数（不足部分0填充），可动态生成文件名（如 000001.png , ...）

/*------------class for accumulator jacobians in parallel--------------*/
class JacobianAccumulator{ // 类似并行的光流，可以并行地计算每个像素点的误差和雅可比
    public:
    JacobianAccumulator( const cv::Mat &img1_,   const cv::Mat &img2_,  
        const VecVector2d &px_ref_,   const vector<double> depth_ref_,   Sophus::SE3d &T21_) :
        img1(img1_),  img2(img2_),  px_ref(px_ref_),  depth_ref(depth_ref_),  T21(T21_)  
        { projection = VecVector2d( px_ref.size(), Eigen::Vector2d(0, 0));
    }

    // accumulate jacobian in a range
    void accumulate_jacobian( const cv::Range &range);
    // get hessian matrix
    Matrix6d hessian() const { return H;  }
    // get bias
    Vector6d bias() const { return b; }
    // get total cost
    double cost_func() const { return cost; }
    // get projected points
    VecVector2d projected_points() const { return projection; }
    // reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

    private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};
/*----------------------class JacobianAccumulator------------*/

/**
 * pose estimation using direct method with multi layer
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,  const cv::Mat &img2,
    const VecVector2d &px_ref,  const vector<double> depth_ref,
    Sophus::SE3d &T21  );

/* pose estimation using direct method with single layer */
void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,   const cv::Mat &img2,
    const VecVector2d &px_ref,   const vector<double> depth_ref,
    Sophus::SE3d &T21 ) ;

// bilinear interpolation
inline float GetPixelValue( const cv::Mat &img, float x, float y ) { // 将函数声明为inline通常需要将函数的定义放在头文件中
    // boundary check
    if (x < 0 ) x = 0;
    if ( y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] + xx * (1- yy) * data[1] 
        + (1 - xx) * yy * data[img.step] + xx * yy * data[img.step + 1]
    );
}

/*-------------------main--------------------*/
// 读取双目的一侧灰度图和双侧视差图，根据多层直接法计算 000001.png - 000005.png 序列图的相机位姿变化，并依次显示金字塔每层的关键点结果
int main(int argc, char **argv) {
    // 读取原始的单侧灰度图和双侧视差图
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // Randomly pick pixels in the first image.
    // Generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    vector<double> depth_ref;

    // generate pixels in ref and load depth data
    for (int i = 0; i < nPoints; i++) { 
        int x = rng.uniform(boarder, left_img.cols - boarder); // don‘t pick pixels close to boarder
        int y = rng.uniform(boarder, left_img.rows - boarder);
        int disparity = disparity_img.at<uchar>(y, x); // 提取随机像素点的视差
        double depth = fx * baseline / disparity; // 计算相应深度信息
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y)); // 随机选取的像素点集，用于直接法运算
    }

    // estimates 01-05.png's pose using this information
    Sophus::SE3d T_cur_ref; // 初始位姿矩阵为单位矩阵

    for (int i = 1; i < 6; i++) { // 1-10
        cv::Mat img = cv::imread( (fmt_others % i).str(), 0); // (fmt_others % i).str() 使用格式化字符串 fmt_others 和整数 i 生成一个文件名，fmt_others 变量在前面已经定义。 0 以灰度模式加载。
        // try single layer by uncomment this line
        // DirectPoseEstimationSingleLayer( left_img, img, pixels_ref, depth_ref, T_cur_ref); // 单层直接法，可用于比较
        DirectPoseEstimationMultiLayer( left_img, img, pixels_ref, depth_ref, T_cur_ref); // 多层直接法
    }
    return 0;
}
/*-------------------main---------------------------------*/

/*----------------------单层直接法--------------------------*/
void DirectiPoseEstimationSingleLayer(
        const cv::Mat &img1,   const cv::Mat &img2,
        const VecVector2d &px_ref,   const vector<double> depth_ref,
        Sophus::SE3d &T21) {
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu( img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_( cv::Range(0, px_ref.size()), // 通过并行运算，计算不同关键点（所选随机像素）的雅可比矩阵
                std::bind( &JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1) );
        Matrix6d H = jaco_accu.hessian(); // hessian() 和 bias() 是类定义的成员函数，用于返回 矩阵H 和 向量b。
        Vector6d b = jaco_accu.bias(); // 根据雅可比计算的成员函数的定义，H 和 b 通过 += 计算（高斯牛顿法增量方程）。尽管并行for将 cv::Range 拆分成多条单线运算，似乎也能通过函数体内的 += 将多线运算的结果合并。

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b); // 求解线性增量方程
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func(); // 这个是返回 cost

        if (std::isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3 ) {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }
    
    cout << "T21 = \n" << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); ++i) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0 ) {
            cv::circle( img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line( img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]), cv::Scalar(0, 250, 0));
        }
    }
    cv::imshow("current", img2_show);
    cv::waitKey();
}
/*----------------------------- void DirectiPoseEstimationSingleLayer -----------------------------------*/

/*----------------------定义计算雅可比矩阵的类成员函数-------------------------*/
void JacobianAccumulator::accumulate_jacobian( const cv::Range &range) {
    // parameters
    const int half_patch_size = 1; // 定义所考虑的像素点周围邻域大小，1即只考虑周围8个像素。同时也是有效像素距图像边界的最短距离。
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i< range.end; i++) {
        // compute the projection in the second image
        Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d( (px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1 );  // (随机关键点)初始像素坐标 -> 初始相机坐标
        Eigen::Vector3d point_cur = T21 * point_ref; // 初始相机坐标 -> 第二相机坐标。T21 为第二相机的位姿。
        if (point_cur[2] < 0)  continue; // depth invalid 深度值负数（相机后面）是无效的。

        float u = fx * point_cur[0] / point_cur[2] + cx,   v = fy * point_cur[1] / point_cur[2] + cy; // 第二像素坐标
        if (u < half_patch_size || u > img2.cols - half_patch_size || 
            v < half_patch_size || v > img2.rows - half_patch_size)  continue;
        
        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0],   Y = point_cur[1],   Z = point_cur[2],
            ZZ = Z * Z,   Z_inv = 1.0 / Z,   Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {
                double error = GetPixelValue(img1,   px_ref[i][0] + x,   px_ref[i][1] + y) // 第二灰度值与第一灰度值的误差
                            - GetPixelValue(img2,   u + x,   v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;
                
                // 第二像素坐标 相对 左扰动delta\xi 的导数
                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                // 第二帧灰度值 相对 像素坐标 的导数（即像素梯度）
                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue( img2,   u + x + 1,   v + y) - GetPixelValue( img2,   u + x - 1,   v + y)), // 此处为关键点向两侧各接一个像素，求解像素梯度。
                    0.5 * (GetPixelValue( img2,   u + x,   v + y + 1) - GetPixelValue( img2,   u + x,   v + y - 1))  );
                
                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose(); 

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error; 
            }
    }

    if (cnt_good) {
        // set hessian, bias and cost
        unique_lock<mutex> lck(hessian_mutex);
        H += hessian; // 当 hessian_mutex 互斥锁在更新全局 H 时被正确锁定时，每个线程都会累积自己的局部 Hessian 矩阵，并在更新全局 H 时进行加锁操作。
        b += bias;
        cost += cost_tmp / cnt_good; // 计算增量方程的H和b的同时，也计算 cost，以判断是否下降。
        // /cnt_good 是为了平均化每个线程计算的误差平方和，因为不同线程处理的有效像素点可能数量不同，通过平均化可获得更稳定的 cost。 （不是所有像素都是有效的）
        // 并行的同步机制：通过互斥锁确保全局变量更新时，没有其他线程同时访问。（虽然多个线程同时计算 cost_tmp 和 cnt_good，但只有一个线程能在任何给定时间更新 cost）
    }
}
/*-----------------------------void JacobianAccumulator::accumulate_jacobian---------------------*/

/*---------------------------------------多层直接法函数-------------------------------------------*/
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,   const cv::Mat &img2,
        const VecVector2d &px_ref,   const vector<double> depth_ref,
        Sophus::SE3d &T21) {
    // parameters 金字塔参数
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // imgae pyramids 金字塔
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else{
            cv::Mat img1_pyr, img2_pyr;
            cv::resize( pyr1[i - 1],   img1_pyr,   // 图像的像素大小缩放
                cv::Size(pyr1[i - 1].cols * pyramid_scale,   pyr1[i - 1].rows * pyramid_scale) );
            cv::resize( pyr2[i - 1],   img2_pyr,   
                cv::Size(pyr2[i - 1].cols * pyramid_scale,   pyr2[i - 1].rows * pyramid_scale) );
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy; // backup the old values
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) { // 遍历关键点的像素坐标集
            px_ref_pyr.push_back(scales[level] * px); // 存储缩放后的关键点像素坐标
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level]; // 根据单目相机模型，像素坐标缩放时，在世界坐标不变的前提下，相机内参也要缩放。
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectiPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21); // 调用单层直接法，实现多层直接法
    }
}