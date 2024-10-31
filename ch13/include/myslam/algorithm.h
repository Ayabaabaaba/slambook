#ifndef MYSLAM_ALGORITHM_H
#define MYSLAM_ALGORITHM_H

#include "myslam/common_include.h"

namespace myslam {
    /**
     * linear triangulation with SVD
     * @param poses   poses,
     * @param points   points in normalized plane
     * @param pt_world   triangulated point in the world
     * @return true if success
     */
    inline bool triangulation( const std::vector<SE3> &poses,  
        const std::vector<Vec3> points,   Vec3 &pt_world ) {  // 三角化得到线性方程组，并通过SVD求解。返回布尔值，表示三角化是否成功
    // 参数：左右目相机位姿（poses，世界系->相机归一化系），归一化平面上的点（points）。输出世界坐标系中的三维路径点（pt_world）
    // 备注：此算法仅在前端中，在当前帧的左右目中匹配一对特征点。（实际上，任意两帧均可三角化）（输入参数均基于当前帧坐标系）
        // 初始化矩阵A和向量b，用于存储线性系统的系数和常数项
        MatXX A(2 * poses.size(), 4 ); // A大小为 2*poses.size() 行（每一帧的位姿可贡献两个方程）和 4 列（三维点按齐次坐标代入）
        VecX b(2 * poses.size() );
        b.setZero();

        for (size_t i = 0; i < poses.size(); ++i ) { // 遍历每个位姿。size_t 是无符号整数类。
            Mat34 m = poses[i].matrix3x4(); // .matrix3x4() 将 SE(3) 转换为 3x4 的投影矩阵（去掉第4行齐次部分）
            // 根据三角化的公式，构建线性方程组 
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0); // .blcok<>() 用于指定子矩阵。<1, 4> 表示希望提取1行4列。(2 * i, 0) 表示子矩阵从原矩阵的第 2 * i 行、第 0 列开始。
            A.block<1, 4>(2 * i + 1, 0 ) = points[i][1] * m.row(2) - m.row(1);
        }

        auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV ); // .bdcSvd() 双向压缩奇异值分解（BD-SVD），()内请求计算薄U和薄V矩阵。
        // 普通奇异值分解（SVD）为截断奇异值分解，通过选择前k个最大的奇异值来压缩。这种压缩是单向的，主要关注于减少矩阵的秩（即非零奇异值的数量）。
        // 双向压缩奇异值分解（BD-SVD）在行和列两个方向上均有压缩，可能导致比截断SVD更大的信息损失，不过存储/计算效率更高。
        pt_world = ( svd.matrixV().col(3) / svd.matrixV()(3, 3) ).head<3>() ;
        // svd.matrixV().col(3)是V矩阵的第四列，即最小奇异值对应的右奇异向量，作为线性方程组的单位化解。
        // 通过除以V矩阵的第4行第4列的元素 svd.matrixV()(3, 3)，将单位解转换为齐次坐标，即三维路标点的齐次坐标。
        // .head<3>()取前三个元素作为三维路标点的世界坐标。

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2 ) { // 检查SVD的奇异值，以评估解的质量。
        // 考虑到齐次坐标的第四维需归一化为1，最小奇异值需要足够小（表明第4维的变化对前三维的世界坐标影响不大）
        // 通过sigma_4 << sigma_3 判断最小奇异值足够小。
            return true;
        }
        return false; // 解质量不好，放弃
    } 

    // converters
    inline Vec2 toVec2( const cv::Point2f p ) { return Vec2(p.x, p.y ); } 
    // 定义一个内联函数toVec2，用于将OpenCV的Point2f类型转换为myslam命名空间下的Vec2类型

} // namespace myslam

#endif // MYSLAM_ALGORITHM_H