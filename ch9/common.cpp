#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.h"
#include "rotation.h"
#include "random.h"

typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
void FscanfOrDie(FILE *fptr, const char *format, T *value) { 
// 从文件指针 fptr 指向的文件中，按 format 格式读取一个值到 value 指向的位置，读取失败则向标准错误输出一条错误信息。 
// T *value 可读取任何类型的数据
    int num_scanned = fscanf(fptr, format, value); // fscanf() 从文件中读取数据，并返回成功读取并赋值的输入项数。
    // fscanf() 函数在成功读取后，会将指针移动到下一个未读取的数据位置。
    if (num_scanned != 1) std::cerr << "Invalid UW data file. \n"; // 如果读取的项数不是1（即没有成功读取到1个值），则输出错误信息。
    // std::cout 为标准输出流（正常的程序输出）。 std::cerr 是标准错误输出流。
}

void PerturbPoint3(const double sigma, double *point) {
// 对一个三维点(point 数组) 的每个坐标加上一个正态分布的随机扰动，扰动标准差为 sigma 指定。
    for (int i = 0; i < 3; ++i)     point[i] += RandNormal() * sigma; // RandNormal() 定义在 random.h 里，生成符合标准正态分布的随机数。
}

double Median(std::vector<double> *data) {
// 返回 data 容器中，中等大小的元素的指针。
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;
    // std::vector<double>::iterator 是 std::vector<double> 容器的一个迭代器类型。迭代器类似于指针的对象，允许按顺序访问容器中的元素，可遍历、修改或访问元素。
    // data->begin() + n / 2 返回 data 中第 n/2 个元素(从0开始)的迭代器。data[0] 访问第一个元素的值，data->begin() 获取指向第一个元素的迭代器（一种特殊指针）。
    std::nth_element(data->begin(), mid_point, data->end());
    // std::nth_element() 重新排列给定范围内的元素，只保证中间参数对应的元素位置正确。
    // data->begin() 和 data->end() 输入范围的起始和结束迭代器。 mid_point 量可换成其他迭代器，这里是使中间大小的元素位于mid_point，比它大的在其后面、比它小的在其前面。
    return *mid_point;
    // 按照函数声明，返回的是double型数据，这里*mid_points是指返回mid_point指针指向的数据。不加*才是返回指针本身。
}

/*---------------BALProblem类的构造函数:不返回任何值（void类型），但会初始化BALProblem对象的状态-----------------*/
BALProblem::BALProblem(const std::string &filename, bool use_quaternions) { // 在类的声明中，如果没有第二个参数输入，则第二个参数默认为false
// 该构造函数接收一个文件名和bool值
    FILE *fptr = fopen(filename.c_str(), "r"); // fopen() C标准库的函数，读取文件。"r" 为只读模式。.c_str() 将std::string 转换为C风格的字符串。
    if (fptr == NULL) { // 如果文件指针为 NULL，则文件打开失败、返回错误信息。
        std::cerr << "Error: unable to open file " << filename;
        return;
    };

    // This will die horribly on invalid files. Them's the breaks. 
    FscanfOrDie(fptr, "%d", &num_cameras_); // 前面自定的函数：读取fptr，按"%d"（表示整数）格式，输入到 &...。读取失败则返回错误信息。
    FscanfOrDie(fptr, "%d", &num_points_); // 按fscanf() 函数的特性，这里依次读取了三个数据进行存放。
    FscanfOrDie(fptr, "%d", &num_observations_);
    // num_cameras_ 相机数量。 num_points_ 3D路标点数量。 num_observations_ 观测到的观测点数量。

    std::cout << "Header: " << num_cameras_ // 输出读取到的头部信息
            << "  " << num_points_
            << "  " << num_observations_;
    
    // 为观测点索引、相机索引和观测2D值分配内存（new 动态分配内存）
    point_index_ = new int[num_observations_]; 
    // 这行代码中，point_index_ 是一个指向 int 类型数组的指针，该数组包含 num_observations_ 个元素。
    // 动态分配的内存，在不再需要时需要释放。在析构函数中有： delete[] point_index_;
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    // 分配存储所有参数（相机参数(6维位姿、1维焦距、2维畸变参数)、点的位置）的内存
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    // 读取每个观测的相机索引、3D点索引和观测值
    for (int i = 0; i < num_observations_; ++i) {
        FscanfOrDie(fptr, "%d", camera_index_ + i);
        FscanfOrDie(fptr, "%d", point_index_ + i);
        for (int j = 0; j < 2; ++j) {
            FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j); // %lf 为double型
        } 
    }

    // 读取所有参数值。BAL后续先依次是所有相机的9维参数，再依次是所有路标点的三维坐标
    for (int i = 0; i < num_parameters_; ++i) {
        FscanfOrDie(fptr, "%lf", parameters_ + i); 
    }

    fclose(fptr); // 关闭文件

    use_quaternions_ = use_quaternions; // 是否使用四元数的标志
    if (use_quaternions) { // 使用四元数时的操作
        // Switch the angle-axis rotations to quaternions.
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_; // 重新分配参数的内存。相机旋转用四元数，故增加一维。
        double *quaternion_parameters = new double[num_parameters_]; // 分配新内存

        // 初始化两个指针，用于遍历原始参数和新的四元数参数。
        double *original_cursor = parameters_; // 原始参数集
        double *quaternion_cursor = quaternion_parameters; // 新四元数参数集

        // 相机旋转向量在前三维，将其转换为四元数，并复制之后的相机参数。
        for (int i = 0; i < num_cameras_; ++i) {
            AngleAxisToQuaternion( original_cursor, quaternion_cursor );
            quaternion_cursor += 4;
            original_cursor += 3;
            for (int j = 4; j < 10; ++j ) { // 复制之后的相机参数
                *quaternion_cursor++ = *original_cursor++; // 先赋值，后自增
            }
        }

        // Copy the rest of the points. 复制空间点的参数
        for (int i = 0; i < 3 * num_points_; ++i) { 
            *quaternion_cursor++ = *original_cursor++;
        }

        // Swap in the quaternion parameters. 释放原始参数数组的内存，并将指针指向新的四元数参数数组。
        delete[]parameters_;
        parameters_ = quaternion_parameters;
    }
}

void BALProblem::WriteToFile( const std::string &filename) const { // 将相关数据写入指定文件。
    FILE *fptr = fopen(filename.c_str(), "w"); // "w" 以写入模式打开文件。
    if (fptr == NULL) {
        std::cerr << "Error: unable to open file " << filename;
        return;
    }

    // fscanf 从文件流中读取数据。 fprintf 从文件流中写入数据。这里的*fptr是指定的文件，和构造函数中的BAL数据不一样。
    fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

    // 遍历所有观测数据，并写入文件。每行包含相机索引、路标点索引和相应的二维观测坐标
    for (int i = 0; i < num_observations_; ++i) {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; ++j) {
            fprintf(fptr, " %g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }

    // 遍历、写入所有相机参数。
    for (int i = 0; i < num_cameras(); ++i) {
        double angleaxis[9]; // double数组，用于存储9维相机参数。
        if (use_quaternions_) { // 采用四元数表示旋转
            // Output in angle-axis format.
            QuaternionToAngleAxis( parameters_ + 10 * i, angleaxis);
            memcpy( angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));
            // memcpy(*dest, *src, n) 从src指向的地址，复制n个字节到dest指向的地址。
            // sizeof(double) 获取double型变量在内存占用的字节数。
            // parameters_ + 10 * i + 4 中的+4，前四项默认为四元数，从第五项开始写入其他数据。
        } else {
            memcpy( angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
        }
        for (int j = 0; j < 9 ; ++j) {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
            // %.16g .16指定有效数字最大数量，%g根据数值大小自动选择定点表示或科学计数法。
        }
    }

    // 写入三维路标点的数据
    const double *points = parameters_ + camera_block_size() * num_cameras_; // 计算指向路标点数据的指针。camera_block_size() 返回相机参数维度，use_quaternions_是成员变量可以不写在参数列表。
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size(); // point_block_size() 返回3（路标点坐标维度）
        for (int j = 0; j < point_block_size(); ++j) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }

    fclose(fptr);
}

// Write the problem to ap PLY file for inspection in Meshlab or CloudCompare
void BALProblem::WriteToPLYFile(const std::string &filename) const { // 将一个包含相机外参和路标3D点的问题写入一个PLY文件，以便meshlab查看点云
    // std::ofstream of(filename.c_str());
    std::ofstream of(filename); // std::ofstream 输出文件流的类，可用于将数据写入文件。创建后，可用各种输出运算符（如<<）向文件写入数据

    // 写入PLY文件的头部信息
    of << "ply" // 文件类型
        << '\n' << "format ascii 1.0" // 文件格式：ASCII编码，版本1,0
        // 定义"vertex"顶点元素，数量为相机数量+路标点数量
        << '\n' << "element vertex " << num_cameras_ + num_points_  
        // 给每个顶点定义属性：x、y、z坐标（float），红绿蓝颜色值（uchar）
        << '\n' << "property float x"
        << '\n' << "property float y"
        << '\n' << "property float z"
        << '\n' << "property uchar red"
        << '\n' << "property uchar green"
        << '\n' << "property uchar blue"
        // 结束头部信息的定义
        << '\n' << "end_header" << std::endl;

    // Export extrinsic data (i.e. camera centers) as green points. 导出相机中心作为绿色点
    double angle_axis[3]; // 存储旋转向量
    double center[3]; // 存储相机中心（平移向量）
    for (int i = 0; i < num_cameras(); ++i) { // 遍历所有相机
        const double *camera = cameras() + camera_block_size() * i; // 各相机参数的起始位置
        CameraToAngleAxisAndCenter(camera, angle_axis, center); // 从相机参数提取旋转向量和平移向量
        of << center[0] << ' ' << center[1] << ' ' << center[2]
            << " 0 255 0" << '\n';
    }

    // Export the structure (i.e. 3D Points) as white points. 导出路标3D点为白点
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); ++i) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); ++j) {
            of << point[j] << ' ';
        }
        of << " 255 255 255\n";
    }

    of.close(); // 关闭文件流
}

void BALProblem::CameraToAngleAxisAndCenter( const double *camera,  double *angle_axis,  double *center) const {
// 从相机参数提取旋转向量和相机中心（不等于平移向量）
    VectorRef angle_axis_ref( angle_axis, 3 );
    if (use_quaternions_) { // 使用四元数时
        QuaternionToAngleAxis( camera, angle_axis );
    } else {
        angle_axis_ref = ConstVectorRef( camera, 3 );
    }

    // c = -R · t  应当理解为 Rc + t = 0
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint( inverse_rotation.data(),  camera + camera_block_size() - 6 , center) ; 
    VectorRef( center, 3) *= -1.0;
}

void BALProblem::AngleAxisAndCenterToCamera( const double *angle_axis, const double *center,  double *camera) const {
// 将旋转向量和相机中心转换回相机参数
    ConstVectorRef angle_axis_ref( angle_axis, 3 );
    if (use_quaternions_) {
        AngleAxisToQuaternion( angle_axis, camera );
    } else {
        VectorRef( camera, 3 ) = angle_axis_ref;
    }

    // t = -R * C
    AngleAxisRotatePoint( angle_axis, center, camera + camera_block_size() - 6 );
    VectorRef( camera + camera_block_size() - 6, 3 ) *= -1.0;
}

void BALProblem::Normalize() { // 对场景归一化处理，使重建的中位数绝对偏差为100（便于点云绘制）
    // Compute the marginal median of the geometry
    std::vector<double> tmp(num_points_); // 定义一个容器变量tmp，容器大小为num_points_
    Eigen::Vector3d median;
    double *points = mutable_points(); // 指针指向三维路标点的起始位置
    for (int i = 0; i < 3; ++i) { // 依次索引和存放x、y、z轴的数据
        for (int j = 0; j < num_points_; ++j) {
            tmp[j] = points[3 * j + i]; 
        }
        median(i) = Median(&tmp); // 分别返回 x、y、z轴的中等大小的数值。（j每次都是从0开始存储）
        // Median() 前面的自定函数，返回容器中的中等大小的数值
    }

    for (int i = 0; i < num_points_; ++i) {
        VectorRef point(points + 3 * i, 3 ); // 从 points + 3 * i 指向的位置开始，读取3个元素存入 point
        tmp[i] = (point - median).lpNorm<1>(); // .lpNorm<1>() 计算L1范数（所有元素绝对值的和）
    }

    const double median_absolute_deviation = Median(&tmp); // 每个路标点与中位数偏差的L1范数，再求中位数。

    // Scale so that the median absolute deviation of the resulting reconstruction is 100.
    const double scale = 100.0 / median_absolute_deviation; // 按比例缩放绝对偏差到100
    // X = scale * (X - median)
    for (int i = 0; i < num_points_; ++i) {
        VectorRef point( points + 3 * i, 3 ); 
        point = scale * (point - median); // 按中位数偏差为100，缩放路标3D点
    }

    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = cameras + camera_block_size() * i; // 每个相机的参数
        CameraToAngleAxisAndCenter( camera, angle_axis, center);
        // center = scale * (center - median)
        VectorRef( center , 3 ) = scale * (VectorRef(center, 3) - median); // 按中位数偏差为100，缩放相机中心。设定center变量为3维，并存放计算后的数据
        AngleAxisAndCenterToCamera( angle_axis, center, camera); // 中位数偏差调整后，重新存放相机参数
    }
}

void BALProblem::Perturb( const double rotation_sigma,   const double translation_sigma,   const double point_sigma) {
// 对路径3D点，以及相机的旋转和平移，添加扰动
    assert( point_sigma >= 0.0);
    assert( rotation_sigma >= 0.0);
    assert(translation_sigma >= 0.0);

    // 给路标3D点添加扰动
    double *points = mutable_points(); // 指向路标点坐标的初始位置
    if (point_sigma > 0) {
        for (int i = 0; i < num_points_; ++i) {
            PerturbPoint3(point_sigma, points + 3 * i); // PerturbPoint3() 前面的自定函数，给三维点的每个坐标，添加一个正态分布的随机扰动，扰动标准差由 sigma 指定
        }
    }

    // 给相机添加旋转和平移扰动
    for (int i = 0; i < num_cameras_; ++i) {
        double *camera = mutable_cameras() + camera_block_size() * i; // 依次访问相机参数

        double angle_axis[3];
        double center[3];
        // Perturb in the rotation of the camera in the angle-axis representation
        // 给旋转添加扰动
        CameraToAngleAxisAndCenter( camera,  angle_axis,  center );
        if (rotation_sigma > 0.0 ) {
            PerturbPoint3( rotation_sigma, angle_axis ); // 直接在旋转向量上加入扰动
        }
        AngleAxisAndCenterToCamera( angle_axis,  center,  camera); 

        // 给平移添加扰动
        if (translation_sigma > 0.0)    PerturbPoint3(translation_sigma,   camera + camera_block_size() - 6 );

    }
}
