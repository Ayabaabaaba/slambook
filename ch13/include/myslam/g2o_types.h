#ifndef MYSLAM_G2O_TYPES_H
#define MYSLAM_G2O_TYPES_H

#include "myslam/common_include.h"
#include <g2o/core/base_binary_edge.h> // 二元边
#include <g2o/core/base_unary_edge.h> // 一元边
#include <g2o/core/base_vertex.h> // 节点
#include <g2o/core/block_solver.h> 
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace myslam 
{
    class VertexPose : public g2o::BaseVertex<6, SE3> { // 位姿顶点
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        virtual void setToOriginImpl() override { _estimate = SE3(); } // 优化变量初始化

        virtual void oplusImpl( const double *update) override { // 优化变量更新
            Vec6 update_eigen;
            update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
            _estimate = SE3::exp( update_eigen ) * _estimate;
        } 

        virtual bool read( std::istream &in ) override { return true; } // 只是单纯模拟读写、返回true，并未进行实际读写。
        virtual bool write( std::ostream &out ) const override { return true; }
    }; //  class VertexPose

    class VertexXYZ : public g2o::BaseVertex<3, Vec3> { // 路标顶点
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        virtual void setToOriginImpl() override { _estimate = Vec3::Zero(); } // 优化变量初始值

        virtual void oplusImpl( const double *update ) override { // 优化变量更新
            _estimate[0] += update[0];
            _estimate[1] += update[1];
            _estimate[2] += update[2];
        }

        virtual bool read( std::istream &in ) override { return true; }
        virtual bool write( std::ostream &out ) const override { return true; }
    }; // class VertexXYZ

    class EdgeProjectionPoseOnly : public g2o::BaseUnaryEdge<2, Vec2, VertexPose > { // 仅估计位姿的一元边。仅在前端中用于位姿估计。
    // 直接法转换为图优化求解，一元是当前帧的相机位姿估计，边是不同路标点在参考帧上的像素灰度观测。
    // <> 参数依次为：观测值维度、观测值类型、该边所连接顶点的类型。
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeProjectionPoseOnly( const Vec3 &pos, const Mat33 &K ) : _pos3d(pos), _K(K) {} // 构造函数。传入三维路标点估计位置、相机内参

        virtual void computeError() override { // 误差计算。路标点的像素投影误差。
            const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);  // static_cast<> 为C++的类型转换操作符，<>内为想转换到的目标类型,()内为被转换的对象。
            // const VertexPose * 即指向 VertexPose 位姿节点的常量指针，即不能通过这个指针修改 VertexPose 对象。
            SE3 T = v->estimate(); // 相机位姿估计值
            Vec3 pos_pixel = _K * (T * _pos3d); // 像素预测坐标。Sophus库对SE3和三维坐标向量的乘法重载了吗？
            pos_pixel /= pos_pixel[2]; // 第三维归一化
            _error = _measurement - pos_pixel.head<2>(); // 观测像素 - 预测像素
            // _measurement 为当前帧在左目的特征点的位置。
            // 在前端的逻辑中，先通过LK光流追踪当前帧上的特征点位置，也即此处的观测值；之后，再通过g2o优化相机位姿（通过相机位姿和已知的三维路标点位置估计当前帧上相应特征点位置）。
            // 该逻辑优化目标是当前帧的特征点位置的误差，并非直接优化灰度值（因为当前帧的正确特征点位置，在前端中通过LK光流追踪已知）
        }

        virtual void linearizeOplus() override { // 雅可比解析式。像素坐标误差（2维）对相机位姿态（6维）的偏导，2*6矩阵。直接法的雅可比矩阵。
            const VertexPose *v = static_cast<VertexPose *>(_vertices[0]);
            SE3 T = v->estimate(); // 相机位姿估计（世界系 -> 当前帧）
            Vec3 pos_cam = T * _pos3d; // 相机系下的三维路标点
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;
            _jacobianOplusXi << -fx * Zinv,   0,            fx * X * Zinv2,   fx * X * Y * Zinv2,        -fx - fx * X * X * Zinv2,   fx * Y * Zinv,
                                0,            -fy * Zinv,   fy * Y * Zinv2,   fy + fy * Y * Y * Zinv2,   -fy * X * Y * Zinv2,        -fy * X * Zinv; 
            // 并非像直接法那样针对灰度值进行优化，而是在已知当前帧特征点正确位置的情况下，针对特征点位置误差进行优化。
        }

        virtual bool read( std::istream &in ) override { return true; }
        virtual bool write( std::ostream &out ) const override { return true; }

        private:
        Vec3 _pos3d;
        Mat33 _K; // 相机内参（前端仅使用左目）
    }; // class EdgeProjectionPoseOnly
    
    class EdgeProjection : public g2o::BaseBinaryEdge<2, Vec2, VertexPose, VertexXYZ> { // 带有地图和位姿的二元边。仅在后端中用于相机位姿和路标点的优化。
        // <> 参数依次为：观测值维度、观测值类型、起点顶点的类型、终点顶点的类型。
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeProjection( const Mat33 &K,  const SE3 &cam_ext ) 
            : _K(K) { _cam_ext = cam_ext;  } // 构造函数。传入左目/右目的内参、位姿（立体视觉 -> 左/右目）

        virtual void computeError() override { // 误差计算。
            const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]); // 相机位姿顶点
            const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]); // 三维路标点顶点
            SE3 T = v0->estimate(); // 相机位姿估计值
            Vec3 pos_pixel = _K * ( _cam_ext * (T * v1->estimate()) ); // 路标点估计值到像素预测坐标
            pos_pixel /= pos_pixel[2]; 
            _error = _measurement - pos_pixel.head<2>(); 
        }

        virtual void linearizeOplus() override { // 雅可比解析式。
            const VertexPose *v0 = static_cast<VertexPose *>(_vertices[0]); // 六维相机位姿
            const VertexXYZ *v1 = static_cast<VertexXYZ *>(_vertices[1]); // 三维路标点
            SE3 T = v0->estimate();
            Vec3 pw = v1->estimate();
            Vec3 pos_cam = _cam_ext * T * pw; // _cam_ext在后端中引入的是左目/右目位姿（立体视觉 -> 左/右目），T为当前帧位姿（世界系 -> 当前帧）
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / ( Z + 1e-18 );
            double Zinv2 = Zinv * Zinv;
            _jacobianOplusXi << -fx * Zinv,   0,            fx * X + Zinv2,   fx * X * Y * Zinv2,        -fx - fx * X * X * Zinv2,   fx * Y * Zinv,
                                0,            -fy * Zinv,   fy * Y * Zinv2,   fy + fy * Y * Y * Zinv2,   -fy * X * Y * Zinv2,        -fy * X * Zinv; // 误差相对第一个顶点（6维相机位姿）的雅可比矩阵（值解法）

            _jacobianOplusXj = _jacobianOplusXi.block<2, 3>(0, 0)  *  _cam_ext.rotationMatrix()  *  T.rotationMatrix(); // 误差相对第二个顶点（三维路标点）的雅可比矩阵。
            // .block<>() <>内为几行几列子矩阵，()为子矩阵的起始位置。
            // 旋转矩阵的右乘。此处对两个旋转矩阵的右乘，将雅可比转换为相对于三维点世界坐标的偏导。
        }

        virtual bool read( std::istream &in ) override { return true; }
        virtual bool write( std::ostream &out ) const override { return true; }

        private:
        Mat33 _K;
        SE3 _cam_ext; 
    }; // class EdgeProjection

} // namespace myslam


#endif // MYSLAM_G2O_TYPES_H