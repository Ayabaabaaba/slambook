#ifndef RAND_H
#define RAND_H // 宏定义

#include <math.h>
#include <stdlib.h>

inline double RandDouble() // 函数返回double型数据，不接受输入参数
{ // 生成一个 [0.0, 1.0) 区间内的均匀分布的随机数
    double r = static_cast<double>(rand());
    // RAND_MAX 为 stdlib.h 头文件中定义的常量。rand() 函数返回 [0, RAND_MAX] 区间内的整数
    // static_cast<double>() 将数据类型转换为double型
    // static_cast 为类型转换操作符，相比于强制转换，在转换不合法时会报错。
    return r / RAND_MAX; // 获取一个 [0.0, 1.0) 之间的随机浮点数。
}

inline double RandNormal() 
{ // 生成一个符合标准正态分布的随机double数据
    double x1, x2, w;
    do{ // 该循环用于拒绝 w>= 1.0 或 w == 0.0（考虑到浮点数的精度，直接==0.0不切合实际） 的组合，这些组合无法映射到有效的正态分布值上。
        x1 = 2.0 * RandDouble() - 1.0;
        x2 = 2.0 * RandDouble() - 1.0; // 生成在 [-1.0, 1.0) 区间内的均匀分布的随机数
        w = x1 * x1 + x2 * x2;
    }while( w >= 1.0 || w == 0.0);

    w = sqrt( (-2.0 * log(w)) / w ); // 应用 Box-Muller 变换的变种，计算变换后的 w 值
    // Box-Muller变换是一种数学方法，可以将两个独立的均匀分布的随机数，转换为两个独立的标准正态分布随机数。
    return x1 * w;
}

#endif // random.h
