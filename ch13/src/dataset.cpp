#include "myslam/dataset.h"
#include "myslam/frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

namespace myslam {
    Dataset::Dataset(const std::string& dataset_path) : dataset_path_(dataset_path) {} // 构造函数，接收一个数据集路径dataset_path

    bool Dataset::Init() { // 用于初始化数据集，返回初始化是否成功。
        // read camera intrinsics and extrinsics
        ifstream fin(dataset_path_ + "/calib.txt"); // 打开数据集路径下的calib.txt文件
        if (!fin) {
            LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!" ;
            return false;
        }

        for (int i = 0; i < 4; ++i ) { // 循环4次，读取4个相机的参数。分别是灰度左右目、彩色左右目
            char camera_name[3]; // 单个相机视角的字符数。比如 P0:
            for (int k = 0; k < 3; ++k ) {
                fin >> camera_name[k]; 
            }
            double projection_data[12];
            for (int k = 0; k < 12; ++k ) { // 读取12个投影参数
                fin >> projection_data[k]; 
            }
            Mat33 K;
            K << projection_data[0], projection_data[1], projection_data[2], // fx, 0, cx
                projection_data[4], projection_data[5], projection_data[6], // 0, fy, cy
                projection_data[8], projection_data[9], projection_data[10]; // 0, 0, 1 // 原始的投影矩阵部分应该是KR，不过这里R=I
            Vec3 t;
            t << projection_data[3], projection_data[7], projection_data[11]; // 原始的投影矩阵部分应该是Kt
            t = K.inverse() * t; 
            K = K * 0.5; // 内参*0.5，应该是后面读取的原始图像尺寸缩放了一半，相应的焦距和光心偏移也缩放一半?
            Camera::Ptr new_camera( new Camera( K(0, 0), K(1, 1), K(0, 2), K(1, 2), t.norm(), SE3(SO3(), t) )  ); // 后两个参数为基线baseline和位姿。
            
            cameras_.push_back(new_camera);
            LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
        }
        fin.close();
        current_image_index_ = 0;
        return true;
    }

    Frame::Ptr Dataset::NextFrame() { // 从数据集中读取下一帧
        boost::format fmt("%s/image_%d/%06d.png"); // 格式化字符串，用于生成图像文件的路径
        cv::Mat image_left, image_right;
        // read images
        image_left = cv::imread( (fmt % dataset_path_ % 0 % current_image_index_).str(), cv::IMREAD_GRAYSCALE);
        image_right = cv::imread( (fmt % dataset_path_ % 1 % current_image_index_).str(), cv::IMREAD_GRAYSCALE); // cv::IMREAD_GRAYSCALE 以灰度模式读取图像
        cout << "读取第 " << current_image_index_ << " 帧数据" << endl;

        if (image_left.data == nullptr || image_right.data == nullptr) { // 如果图像读取失败，记录警告日志并返回nullptr
            LOG(WARNING) << "cannot find images at index " << current_image_index_;
            return nullptr;
        }

        cv::Mat image_left_resized, image_right_resized; // 将左右图像缩小到原来的一半，通过降低图像分辨率来加快处理速度，但也会降低图像精度
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST); // cv::Size() 用于指定图像大小，如果不指定新的大小（cv::Size()）则需指定缩放因子（此处为0.5），以便按比例缩放图像。
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST); // cv::INTER_NEAREST 用于图像缩放的最简单的插值方法，选择最接近目标像素的输入像素值作为输出像素值。计算速度快，但可能产生块状效果（忽略了像素之间的平滑过渡）。

        auto new_frame = Frame::CreateFrame(); // 将缩小后的图像赋值给新的Frame对象，增加图像索引，并返回这个Frame对象。
        new_frame->left_img_ = image_left_resized; 
        new_frame->right_img_ = image_right_resized;
        current_image_index_++;
        return new_frame;
    }
} // namespace myslam