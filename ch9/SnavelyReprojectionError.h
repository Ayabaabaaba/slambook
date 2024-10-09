#ifndef SnavelyReprojection_H 
#define SnavelyReprojection_H
// 宏定义，可避免绝大多数重定义错误
// #endif // 放于.h末尾

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError { // 投影误差模型
    public:
    SnavelyReprojectionError( double observation_x, double observation_y ) : observed_x(observation_x), observed_y(observation_y) {} // 构造函数

    /*---- 操作符 ()，一个模板函数，这里重载用于计算重投影误差。两个输入指针 camera (指向相机参数的数组) 和 point (指向三维点坐标的数组)，一个输出数组 residuals (存储计算出的残差) ------*/
    template< typename T >
    bool operator() (const T *const camera, const T *const point, T *residuals) const {
    // 外部调用方式：先创建实例 SnavelyReprojectionError Exm; 再通过()运算： Exm(camera, point, residuals);
        // camera[0, 1, 2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion( camera, point, predictions); // 世界系3D坐标 -> 第二相机预测像素坐标
        residuals[0] = predictions[0] - T(observed_x); // 预测 与 观测 的残差
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation 轴-角旋转（即旋转向量）
    // [3-5] : translation
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T> 
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) { // 定义一个静态模板函数 CamProjectionWithDistortion()
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint( camera, point, p); // 求解旋转后的点，默认只读取camera前三位。世界系 -> 相机系（不含平移）
        // camera[3, 4, 5] are the translation
        p[0] += camera[3]; // 相机系坐标（平移修正）
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center for distortion
        T xp = -p[0] / p[2]; // 相机归一化坐标
        T yp = -p[1] / p[2];

        // Apply second and forth order radial distortion
        const T &l1 = camera[7]; // 只含径向畸变的二阶项和四阶项
        const T &l2 = camera[8];

        T r2 = xp * xp + yp * yp; 
        T distortion = T(1.0) + r2 * (l1 + l2 * r2); // 去畸变的系数

        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp; // 相机归一化坐标 -> 去畸变坐标 -> 像素坐标
        predictions[1] = focal * distortion * yp;

        return true;
    }

    /*------ 静态成员函数 Create() ，生成并返回一个 ceres::CostFunction 对象的指针 -----------*/
    static ceres::CostFunction *Create( const double observed_x, const double observed_y ) { // 输入量为二维观测值
    // Static 意味着 Create() 属于类本身（不是类的某个实例），可通过类名直接调用（无需创建类的实例）
        return ( new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3> (
            new SnavelyReprojectionError(observed_x, observed_y)  )     );
        // new 动态分配内存。
        // 创建 SnavelyReprojectionError 类的实例，2为代价函数输出的残差维度，9为待优化变量(相机参数)的维度，3为输入参数(三维点坐标)维度。
        // ceres::AutoDiffCostFunction 的构造函数接收一个指向 SnavelyReprojectionError 实例的指针，该实例用于计算残差
        // SnavelyReprojectionError(observed_x, observed_y) 这里是按该实例的构造函数运行？
    }
    // 类本身是一种抽象的数据类型，不占用内存空间，只描述对象的创建方式和可执行的操作。
    // 类的实例是根据类的定义，创建的具体对象，每个实例都有独立的内存空间来存储自己的数据成员的值。实例是类的具体化。
    // 类的静态成员函数属于类本身，而不属于类的某个实例，可以通过类名直接调用而无需创建类的实例。（不依赖类的实例状态，因为不访问类的非静态成员变量）可以不inline放在头文件中，但如果频繁被多个.cpp调用，仍建议inline
    // 静态成员函数不直接访问非静态成员，但可以通过一个类的实例调用非静态成员函数，从而间接访问实例的非静态成员变量。静态成员函数也可在被调用时，接收外部参数。
    // 静态成员变量需要static声明，属于类本身而非类的实例，所有类的实例都可访问，且访问同一静态成员变量的副本（需注意是否冲突）

    private:
    double observed_x;
    double observed_y;
};

#endif // 宏定义结尾，SnavelyReprojection.h