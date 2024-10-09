#ifndef ROTATION_H
#define ROTATION_H
// 宏定义，可避免绝大多数重定义错误
// #endif // 放于.h末尾

#include <algorithm>
#include <cmath>
#include <limits>

/*--- math functions needed for rotation conversion ---*/
// dot and cross production
template<typename T> // 模板声明，typename 同 class 一样为通用类型。T为自定义的模板参数，只需确保函数体内的类型统一即可。
inline T DotProduct(const T x[3], const T y[3]) { // 三维向量点乘
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template<typename T> // 不同的函数模板的定义，都需要独立的模板声明。即使与前面用的相同的T，两个函数体的参数返回类型都是独立的（也可写为不同的模板参数）
inline void CrossProduct(const T x[3], const T y[3], T result[3]) { // 三维向量叉乘
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0];
}

/*-----------------------------------------------------*/
template<typename T>
inline void AngleAxisToQuaternion(const T *angle_axis, T *quaternion) { // 旋转向量 -> 四元数
    const T &a0 = angle_axis[0];
    const T &a1 = angle_axis[1];
    const T &a2 = angle_axis[2]; // 这里的向量是包含旋转角度theta的，不是单位长度的旋转轴。
    const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2; 

    if (theta_squared > T(std::numeric_limits<double>::epsilon())) {
    // std::numeric_limits<double>::epsilon() 为C++标准库的函数，返回double类型能表示的最小正数（double类型精度的下限）。T(·) 表示类型转换操作。
    // 本质就是判断 > 0
        const T theta = sqrt(theta_squared);
        const T half_theta = theta * T(0.5);
        const T k = sin(half_theta) / theta; 
        quaternion[0] = cos(half_theta);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
    else { // in case if theta_squared is zero
        const T k(0.5); // 旋转角度为0时，没有旋转，k可取任意数值（单纯确保数据稳定）
        // 和 const T k = 0.5 或 const T k = T(0.5) 等价，0.5本身即是可隐式转换类型的字面量。
        quaternion[0] = T(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

template<typename T>
inline void QuaternionToAngleAxis(const T *quaternion, T *angle_axis) { // 四元数 -> 旋转向量
    const T &q1 = quaternion[1];
    const T &q2 = quaternion[2];
    const T &q3 = quaternion[3];
    const T sin_squared_half_theta = q1 * q1 + q2 * q2 + q3 * q3; // 引用的变量不需要再加&

    // For quaternions representing non-zero rotation, the conversion is numercially stable.
    if (sin_squared_half_theta > T(std::numeric_limits<double>::epsilon())) {  // 判断 > 0
        const T sin_half_theta = sqrt(sin_squared_half_theta);
        const T &cos_half_theta = quaternion[0];

        // If cos_half_theta < 0, then half_theta > pi/2, which means that theta > pi. 
        const T theta = T(2.0) * 
            ((cos_half_theta < 0.0) ? atan2(-sin_half_theta, -cos_half_theta) : atan2(sin_half_theta, cos_half_theta)); // T(2.0) 为强制类型转换。
            // <条件> ? <表达式1> : <表达式2>   C++的三元条件运算符，若条件为真，则表达式1；为假则表达式2.
            // atan2(y,x) 返回的值在 (-pi,pi)，根据(y,x)所在象限返回角度
            // 这里确保旋转角度 theta 在 -pi~pi之间取值。如果 cos_half_theta < 0，直接 atan2(sin_half_theta, cos_half_theta) 会使 theta/2 > pi/2，进而 theta > pi.
            // acos和asin的值域无法包含整个-pi~pi，并且在取边界1时，结果会不稳定。故使用atan2()求解。
        const T k = theta / sin_half_theta; // 这里不单独返回旋转角度，将theta乘入旋转轴即包含完整的旋转信息。

        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }
    else { // 判断 = 0（平方和不可能 < 0）
        // For zero rotation, sqrt() will produce NaN in derivative since the argument is zero. ?
        // By approximating with a Taylor series, and truncating at one term, the value and first derivatives will be computed correctly when Jets are used. ?
        const T k(2.0); 
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k; 
    }
}

/*-------------------------------------------------*/
template<typename T>
inline void AngleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3]) { // 通过旋转向量，计算旋转后的点
    const T theta2 = DotProduct(angle_axis, angle_axis); // 点乘提取旋转角度的平方 theta^2
    if (theta2 > T(std::numeric_limits<double>::epsilon())) { // 判断 > 0
        // Away from zero, use the rodrigues formula.
        const T theta = sqrt(theta2); // 旋转角度
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = 1.0 / theta;

        const T w[3] = {angle_axis[0] * theta_inverse, // 提取出旋转轴（单位向量）
                    angle_axis[1] * theta_inverse,
                    angle_axis[2] * theta_inverse};
        
        T w_cross_pt[3];
        CrossProduct(w, pt, w_cross_pt); // Cross Product. 旋转轴 叉乘 目标点 

        const T tmp = DotProduct(w, pt) * (T(1.0) - costheta); // Dot Product * (1 - cos_theta)

        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    }
    else {
        // Near zero, the frist order Taylor approximation of the rotation matrix R corresponding to a vector and angle is 
        //     R = I + hat(w) * sin(theta)
        // But sin(theta) ~ theta and theta * w = angle_axis, which gives us
        //     R = I + hat(w)
        // and actually performing multiplication with the point pt, gives us 
        //     R * pt = pt + w X pt
        // Switching to the Taylor expansion near zero provides meaningful derivatives when evaluated using Jets.
        T w_cross_pt[3];
        CrossProduct(angle_axis, pt, w_cross_pt);

        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}

#endif // 宏定义结尾
