#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

#include <boost/timer.hpp> // 计时器功能

#include "sophus/so3.hpp"
#include "sophus/se3.hpp"
using Sophus::SE3d;

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
using namespace Eigen;  // Eigen库里有和Sophus库重名的函数，不太建议去掉命名空间

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

/**********************************************
* 单目相机在已知轨迹下的稠密深度估计
* 极线搜索 + NCC 匹配
* 程序可以进一步改进
***********************************************/

/*-------------- parameters -------------*/
const int boarder = 20;         // 边缘宽度
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度

const double fx = 481.2f;       // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;

const int ncc_window_size = 3;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积

const double min_cov = 0.1;     // 深度收敛判定：最小方差
const double max_cov = 10;      // 发散判定：最大方差
/*---------------------------------------*/

/*------------函数声明-----------------*/
bool readDatasetFiles(  // 从 REMODE数据集 读取数据
    const string &path,   vector<string> &color_image_files,   
    vector<SE3d> &poses,   cv::Mat &ref_depth);

/**
 * 根据新的图像更新深度估计
 * @param ref 参考图像
 * @param curr 当前图像
 * @param T_C_R 参考图像到当前图像的位姿
 * @param depth 深度
 * @param depth_cov 深度方差
 * @return 是否成功
 */
bool update( 
    const Mat &ref,   const Mat &curr,
    const SE3d &T_C_R,   Mat &depth,   Mat &depth_cov2  );

/**
 * 极线搜索
 * @param ref 参考图像
 * @param curr 当前图像
 * @param T_C_R 位姿
 * @param pt_ref 参考图像中点的位置
 * @param depth_mu 深度均值
 * @param depth_cov 深度方差
 * @param pt_curr 当前点
 * @param epipolar_direction 极线方向
 * @return 是否成功
 */
bool epipolarSearch( 
    const Mat &ref,   const Mat &curr,   const SE3d &T_C_R,
    const Vector2d &pt_ref,   const double &depth_mu,   const double &depth_cov,
    Vector2d &pt_curr,   Vector2d &epipolar_direction  );

/**
 * 更新深度滤波器
 * @param pt_ref 参考图像点
 * @param pt_curr 当前图像点
 * @param T_C_R 位姿
 * @param epipolar_direction 极线方向
 * @param depth 深度均值
 * @param depth_cov2 深度方向
 * @return 是否成功
 */
bool updateDepthFilter(
    const Vector2d &pt_ref,   const Vector2d &pt_curr,   const SE3d &T_C_R,
    const Vector2d &epipolar_direction,   Mat &depth,   Mat &depth_cov2   );

/**
 * 计算 NCC 评分
 * @param ref 参考图像
 * @param curr 当前图像
 * @param pt_ref 参考点
 * @param pt_curr 当前点
 * @return NCC评分
 */
double NCC(
    const Mat &ref,   const Mat &curr,  
    const Vector2d &pt_ref,   const Vector2d &pt_curr  );

// 双线性灰度插值
inline double getBilinearInterpolatedValue( const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int( pt(1, 0) ) * img.step + int( pt(0, 0) )];
    // img.data 为指向图像数据首字节的指针（灰度图像通常为一个连续的uchar数组）。img.step 为图像中一行所占的字节数，通常为图像宽度乘以每个像素所占字节数。
    double xx = pt(0, 0) - floor( pt(0, 0) ); // .floor() 为向下取整。此处计算像素值的小数部分，作为相邻像素的权重。
    double yy = pt(1, 0) - floor( pt(1, 0) );
    return ( (1 - xx) * (1 - yy) * double( d[0] )  // 四个相邻像素(x,y)、(x+1,y)、(x,y+1)、(x+1,y+1)的加权平均 
        + xx * (1 - yy) * double( d[1] ) 
        + (1 - xx) * yy * double( d[img.step] ) 
        + xx * yy * double( d[img.step + 1] )) / 255.0; // 最后的255.0用于归一化。uchar值的范围是[0,255]
}

// --------------------------------------------------------------------
// 一些小工具
// 显示估计的深度图
void plotDepth( const Mat &depth_truth,   const Mat &depth_estimate );

// 像素坐标 -> (去畸变)相机归一化坐标
inline Vector3d px2cam( const Vector2d px ) {
    return Vector3d(
        ( px(0, 0) - cx) / fx,    ( px(1, 0) - cy) / fy,    1
    );
}

// (去畸变)相机坐标(可非归一化) -> 像素坐标
inline Vector2d cam2px( const Vector3d p_cam ) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

// 检测到一个点是否在图像边框内
inline bool inside( const Vector2d &pt ) {
    return pt(0, 0) >= boarder   &&   pt(1, 0) >= boarder
        &&   pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height; 
}

// 显示极线匹配
void showEpipolarMatch( 
    const Mat &ref,   const Mat &curr,
    const Vector2d &px_ref,   const Vector2d &px_curr);

// 显示极线
void showEpipolarLine(
    const Mat &ref,   const Mat &curr, 
    const Vector2d &px_ref,   const Vector2d &px_min_curr, 
    const Vector2d &px_max_curr  );

// 评测深度估计
void evaludateDepth( const Mat &depth_truth,   const Mat &depth_estimate );
// ---------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: dense_mapping path_to_test_dataset" << endl;
        return -1;
    }

    // 从数据集读取数据
    vector<string> color_image_files; // 存放每个图像文件的路径
    vector<SE3d> poses_TWC; // 存放每个图像对应的相机外参(位姿)
    Mat ref_depth; // 单个Mat对象，存放所有图像的depth矩阵数据
    bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    // 第一张图
    Mat ref = imread(color_image_files[0], 0);    // 第一张图像读取作为参考。第二个参数 0 表示以灰度模式读取
    SE3d pose_ref_TWC = poses_TWC[0];   // 第一张图像对应的相机位姿
    double init_depth = 3.0;    // 深度初始值（可自定义，可能影响结果的收敛）
    double init_cov2 = 3.0;     // 方差初始值
    Mat depth(height, width, CV_64F, init_depth);             // 深度图（创建Mat对象）。CV_64F表示64位浮点数，即double型。
    Mat depth_cov2(height, width, CV_64F, init_cov2);         // 深度图方差
    for (int index = 1; index < color_image_files.size(); index++) { // 遍历剩余的图像
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0); // 灰度模式读取当前图像
        if (curr.data == nullptr) continue; // 如果读取失败，则continue跳过当前循环
        SE3d pose_curr_TWC = poses_TWC[index]; // 读取当前相机位姿
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;   // 变换矩阵的坐标系转换，参考帧R到当前帧C的变换： T_{CR} = T_{CW} * T_{WR} . 用作图像预处理？
        update(ref, curr, pose_T_C_R, depth, depth_cov2); // 更新深度估计
        evaludateDepth(ref_depth, depth); // 评测深度估计
        plotDepth(ref_depth, depth); // 显示估计的深度图
        imshow("image", curr); // 显示当前帧图像
        waitKey(1); // 等待1ms以响应键盘事件。1ms无按下则返回-1,继续执行后续程序（但是会提示无响应，怎么解决？）。设置为0则会无期限等待。
    }

    cout << "estimation returns, saving depth map ... " << endl;
    imwrite("./dense_mono/build/depth.png", depth); // 存储最后的深度图。如果目录不存在就不会保存，不会自动创建目录，也不会报错（可能需要额外代码）
    cout << "done." << endl;

    return 0;
}

bool readDatasetFiles(
    const string &path,    vector<string> &color_image_files,
    std::vector<SE3d> &poses,   cv::Mat &ref_depth){
        
    ifstream fin( path + "/first_200_frames_traj_over_table_input_sequence.txt" ); // 打开轨迹文件（位姿）
    if (!fin)   return false;

    while (!fin.eof()) { // 循环读取文件，直到文件末尾
        // 数据格式：图像文件名, tx, ty, tz, qx, qy, qz, qw。 注意是 TWC 而非 TCW
        string image;
        fin >> image; // 读取图像文件名
        double data[7];
        for (double &d:data)   fin >> d; // 读取位姿

        color_image_files.push_back( path + string("/images/") + image); // 存储图像文件的路径
        poses.push_back( // 从读取的位姿数据构造一个SE3d对象，并存放到 poses
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good())   break; // 判断读取数据时遇到错误（如，到达文件末尾后的额外读取尝试）
    }
    fin.close();

    // load reference depth 加载参考深度图
    fin.open( path + "/depthmaps/scene_000.depth" ); // 打开深度图文件
    ref_depth = cv::Mat( height, width, CV_64F ); // CV_64F表示64位浮点数double型
    // cv::Mat 用于存储图像、矩阵等数据的类。输入参数：height为矩阵(或图像)的高度，即行数；width为矩阵(或图像)的宽度，即列数；CV_64F矩阵中元素的数据类型。
    if (!fin)   return false;
    for (int y = 0; y < height; y++ )
        for (int x = 0; x < width; x++ ) {
            double depth = 0 ;
            fin >> depth; // 从文件读取一个深度值
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
            // .ptr<>() 为 cv::Mat 类的一个模板成员函数，用于获取指向矩阵中特定行的指针，允许直接访问矩阵的底层数据并以指针形式操作。
            // .ptr<double>(y) 获取第y行的指向double型数据的指针。指针可以当作数组 [x] 获取第x个元素。
        }
    
    return true;
}

// 对整个深度图进行更新
bool update( const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2 ) {
// ref 参考帧。curr 当前帧。T_C_R 相机参考帧到当前帧的变化矩阵。要更新的深度图 depth 和深度协方差图 depth_cov2
    for (int x = boarder; x < width - boarder; x++ ) 
        for ( int y = boarder; y< height - boarder; y++ ) { // 嵌套循环遍历深度图中的每个像素。实际可以改为GPU加速
            // 遍历每个像素
            if (depth_cov2.ptr<double>(y)[x] < min_cov  // 深度收敛或发散。
                || depth_cov2.ptr<double>(y)[x] > max_cov)    continue; // 检查当前像素的深度协方差值是否在有效范围内，若不在就跳过。
            // 在极线上搜索 (x,y) 的匹配
            Vector2d pt_curr; // 用于存储当前帧的匹配点
            Vector2d epipolar_direction; // 用于存储当前帧的极线方向
            bool ret = epipolarSearch( ref,   curr,   T_C_R,   Vector2d(x, y), 
                depth.ptr<double>(y)[x],   sqrt(depth_cov2.ptr<double>(y)[x]), 
                pt_curr,   epipolar_direction  );  // 在极线上搜索，与参考帧像素(x,y)对应的匹配点
            
            if (ret == false)   continue;   //匹配失败

            // 显示匹配
            showEpipolarMatch( ref,  curr,  Vector2d(x, y),  pt_curr); // 显示参考帧和当前帧的匹配点

            // 匹配成功，更新深度图
            updateDepthFilter( Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2 );
        }
}

// 极线搜索
bool epipolarSearch( const Mat &ref,  const Mat &curr,  const SE3d &T_C_R, 
    const Vector2d &pt_ref,  const double &depth_mu,  const double &depth_cov,
    Vector2d &pt_curr,  Vector2d &epipolar_direction ) {
// ref 参考帧。curr 当前帧。T_C_R 参考帧到当前帧的变换矩阵。pt_ref 参考帧中的像素点，以及该点的深度均值 depth_mu 和深度协方差 depth_cov。输出：当前帧对应的像素点 pt_curr 和极线方向 epipolar_direction.

    Vector3d f_ref = px2cam(pt_ref);  // 像素坐标 -> 已去畸变的相机归一化坐标
    f_ref.normalize(); // 注意：此处的深度指从相机坐标系原点O1到三维点的距离O1P，不是z的坐标，所以这里需要单位化。
    Vector3d P_ref = f_ref * depth_mu; // 参考帧的 O_1P 向量。深度均值即最优估计.
    Vector2d px_mean_curr = cam2px(T_C_R * P_ref);  // 按深度均值投影的像素。 cam2px() 去畸变的相机坐标 -> 像素坐标

    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;  // 计算深度搜索范围的最小值和最大值（均值±3σ），并确保最小值不小于0.1
    if (d_min < 0.1)  d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));   // 按最小深度投影的像素
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));   // 按最大深度投影的像素

    Vector2d epipolar_line = px_max_curr - px_min_curr;   // 极线（线段形式）
    epipolar_direction = epipolar_line;    // 极线方向（当前帧，二维）
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();   // 极线线段的半长度
    if (half_length > 100)   half_length = 100;  // 我们不希望搜索太多东西

    // 显示极线（线段）
    showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

    // 在极线上搜索匹配点。以深度均值点px_mean_curr为中心，左右各取半长度half_length，对极线上的每个点px_curr检查是否在图像内，并计算NCC值。
    double best_ncc = -1.0; // 随便写的一个尽可能小的初始值
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7 ) { // l += sqrt(2)
        Vector2d px_curr = px_mean_curr + l * epipolar_direction; // 待匹配点
        if (!inside(px_curr))      continue; // inside() 自定义函数，判定像素是否图像边界内。
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }  

    if (best_ncc < 0.85f)   return false;  // 若最高的NCC仍低于阈值0.85,则认为匹配失败。
    pt_curr = best_px_curr;
    return true;
}

double NCC( const Mat &ref,   const Mat &curr,
    const Vector2d &pt_ref,   const Vector2d &pt_curr) {
// ref 参考帧。curr 当前帧。pt_ref 参考帧像素。pt_curr 当前帧像素。

    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr;  // 参考帧和当前帧的均值
    for ( int x = -ncc_window_size; x <= ncc_window_size; x++ ) 
        for ( int y = -ncc_window_size; y <= ncc_window_size; y++) { // 分别沿横向和纵向遍历小块内的像素
            double value_ref = double( ref.ptr<uchar>( int(y + pt_ref(1, 0)) )[ int( x + pt_ref(0, 0))]) / 255.0; // main函数里为灰度读取帧。像素归一化处理有助于减少溢出风险。
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue( curr, pt_curr + Vector2d(x, y)); // 从参考帧选取的像素坐标为整定的，但当前帧比较的像素坐标不一定是整定的。
            mean_curr += value_curr;

            values_ref.push_back( value_ref );
            values_curr.push_back( value_curr );
        }
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    // 计算 Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for ( int i = 0; i < values_ref.size(); i++) {
        double n = ( values_ref[i] - mean_ref ) * ( values_curr[i] - mean_curr );
        numerator += n; // numerator 分子
        demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref); // denominator 分母
        demoniator2 += ( values_curr[i] - mean_curr ) * ( values_curr[i] - mean_curr ); 
    }
    return numerator / sqrt( demoniator1 * demoniator2 + 1e-10 ); // 防止分母出现零
}

bool updateDepthFilter( const Vector2d &pt_ref,   const Vector2d &pt_curr,
    const SE3d &T_C_R,   const Vector2d &epipolar_direction,   Mat &depth,   Mat &depth_cov2) {
// pt_ref 参考帧的像素点。pt_curr 当前帧的像素点。T_C_R 参考帧到当前帧的变换矩阵。epipolar_direction 当前帧投影的极线方向。深度图 depth 和深度协方差图 depth_cov2 （囊括更新前后）。

    // 用三角化计算深度
    SE3d T_R_C = T_C_R.inverse(); // 当前帧 -> 参考帧
    // px2cam() 像素坐标 -> 去畸变的相机归一化坐标
    Vector3d f_ref = px2cam(pt_ref); // 参考帧上的像素点
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr); // 当前帧上的像素点
    f_curr.normalize();

    // 方程：同一三维点在两个帧坐标系下的表示，通过旋转和平移构建联系。
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC // 这是包含了两帧各自深度信息的旋转和平移
    // f2 = R_RC * f_cur // 当前帧的深度单位向量，旋转到参考帧坐标系下
    // 转化成下面这个矩阵方程组
    // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //    [ f_2^T f_ref, -f2^T f2      ] [d_cur] = [f2^T t   ]
    Vector3d t = T_R_C.translation(); // 当前帧到参考帧的平移向量
    Vector3d f2 = T_R_C.so3() * f_curr; // 将当前帧的深度单位向量，旋转到参考帧坐标系下。
    Vector2d b = Vector2d( t.dot(f_ref), t.dot(f2) ); // 分别求两个点积，作为矩阵方程的右边向量b
    Matrix2d A; // 计算左边的系数矩阵A
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
    Vector2d ans = A.inverse() * b; // 解线性方程组，获得深度信息。
    // 对两个深度信息进行融合
    Vector3d xm = ans[0] * f_ref;  // 参考帧坐标系下，从参考帧深度信息得到的三维点坐标。
    Vector3d xn = t + ans[1] * f2; // 参考帧坐标系下，从当前帧深度信息得到的三维点坐标。
    Vector3d p_esti = (xm + xn) / 2.0;  // 取两者的平均作为三维点P的位置估计（参考帧坐标系下）
    double depth_estimation = p_esti.norm();  // 深度值

    // 计算不确定性（以一个像素为误差）
    // 加入扰动前的几何关系
    Vector3d p = f_ref * depth_estimation;
    Vector3d a = p - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos( f_ref.dot(t) / t_norm );
    double beta = acos( -a.dot(t) / (a_norm * t_norm) );
    // 扰动一个像素后的几何关系
    Vector3d f_curr_prime = px2cam( pt_curr + epipolar_direction ); // 扰动n个像素：pt_curr + n * epipolar_direction
    f_curr_prime.normalize();
    double beta_prime = acos( f_curr_prime.dot(-t) / t_norm );
    double gamma = M_PI - alpha - beta_prime;
    // 正弦定理得到扰动像素后的当前帧深度信息
    double p_prime = t_norm * sin(beta_prime) / sin(gamma) ;
    // 扰动引起的深度不确定性（方差）
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    // 高斯融合
    // 参考帧的原深度和原深度方差（也可为上一步更新后的）
    double mu = depth.ptr<double>( int(pt_ref(1, 0)) )[ int(pt_ref(0, 0)) ]; 
    double sigma2 = depth_cov2.ptr<double>( int( pt_ref(1, 0) ) )[ int(pt_ref(0, 0)) ]; 
    // 使用高斯融合公式更新深度估计和不确定性（方差）
    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2 ) / (sigma2 + d_cov2); 
    // 更新深度图和深度图的不确定性图
    depth.ptr<double>( int(pt_ref(1, 0)) )[ int(pt_ref(0, 0)) ] = mu_fuse;
    depth_cov2.ptr<double>( int(pt_ref(1, 0)) )[ int(pt_ref(0, 0)) ] = sigma_fuse2;

    return true; 
}

void plotDepth( const Mat &depth_truth, const Mat &depth_estimate ) { // 显示真实深度图、估计深度图和深度误差图
    imshow("depth_truth", depth_truth * 0.4); // 乘以0.4是为了可视化调整亮度 
    imshow("depth_estimate", depth_estimate * 0.4);
    imshow("depth_error", depth_truth - depth_estimate);  // 真实深度与估计深度的差异值
    waitKey(1);
}

void evaludateDepth( const Mat &depth_truth,   const Mat &depth_estimate ) { // 评估深度估计的准确性，计算平均误差和平均平方误差
    double ave_depth_error = 0;  // 初始化平均误差
    double ave_depth_error_sq = 0;   // 初始化平方误差
    int cnt_depth_data = 0; // 初始化有效深度数据计数
    for (int y = boarder; y < depth_truth.rows - boarder; y++ )
        for (int x = boarder; x < depth_truth.cols - boarder; x++ ) { // 遍历每个像素
            double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x]; // 计算真实深度和估计深度的误差
            ave_depth_error += error; // 累积误差
            ave_depth_error_sq += error * error; // 累积平方误差
            cnt_depth_data++; // 数据计数
        }
    ave_depth_error /= cnt_depth_data; // 平均误差
    ave_depth_error_sq /= cnt_depth_data;

    cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}

void showEpipolarMatch( const Mat &ref,  const Mat &curr,  const Vector2d &px_ref,  const Vector2d &px_curr ) { // 在极线匹配中，显示参考帧和当前帧的对应点。
    Mat ref_show, curr_show; // 用于显示的图像副本
    cv::cvtColor( ref, ref_show, cv::COLOR_GRAY2BGR); // 将灰度图像转换为BGR格式（方便显示线条颜色）
    cv::cvtColor( curr, curr_show,cv::COLOR_GRAY2BGR );
    
    // 在参考帧和当前帧绘制对应点，使用蓝色圆表示
    cv::circle( ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2 ); 
    cv::circle( curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2 );

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1); // 等待1ms以显示更新
}

void showEpipolarLine( const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr, const Vector2d &px_max_curr ) { // 在极线匹配中，显示参考帧的点和当前帧的极线及其端点
    Mat ref_show, curr_show; // 创建用于显示的图像副本  
    cv::cvtColor( ref, ref_show, cv::COLOR_GRAY2BGR ); // 将灰度图像转换为BGR格式  
    cv::cvtColor( curr, curr_show, cv::COLOR_GRAY2BGR);

    // 在参考帧绘制参考点，使用绿色圆表示
    cv::circle( ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    // 在当前帧绘制极线的两个端点，使用绿色圆表示  
    cv::circle( curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle( curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    // 在当前帧绘制极线，使用绿色线表示 
    cv::line( curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1); // 等待1毫秒以更新显示  
}