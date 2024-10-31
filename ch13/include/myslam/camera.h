#pragma once

#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"

namespace myslam { // 定义一个命名空间 myslam。可防止命名冲突。
// 命名空间内部定义的所有类型、函数、变量等，只能在命名空间内访问，或通过 myslam::Camera 等访问。

    // Pinhole stereo camera model
    class Camera { // Camera类用于表示相机模型。
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Camera> Ptr; 

        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0; // Camera intrinsics 内参
        SE3 pose_;  // extrinsics, from stereo camera to single camera. 外参。立体相机->单目相机。 // 双目相机中的立体视觉依照左右图视差进行构建。
        SE3 pose_inv_;   // inverse of extrinsics

        Camera(); // 构造函数

        Camera(double fx, double fy, double cx, double cy, double baseline, const SE3 &pose):
            fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) 
            {  pose_inv_ = pose_.inverse();  }

        SE3 pose() const { return pose_; }

        // return intrinsic matrix
        Mat33 K() const { // 内参矩阵
            Mat33 k;
            k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
            return k;
        }

        // coordinate transform: world, camera, pixel. 类成员函数的参数列表，通常为外部输入参数。
        Vec3 world2camera( const Vec3 &p_w, const SE3 &T_c_w );
        Vec3 camera2world( const Vec3 &p_c, const SE3 &T_c_w );
        Vec2 camera2pixel( const Vec3 &p_c );
        Vec3 pixel2camera( const Vec2 &p_p, double depth = 1 );
        Vec3 pixel2world( const Vec2 &p_p, const SE3 &T_c_w, double depth = 1 );
        Vec2 world2pixel( const Vec3 &p_w, const SE3 &T_c_w );
    };
}  // namespace myslam

#endif // MYSLAM_CAMERA_H