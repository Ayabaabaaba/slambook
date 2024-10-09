#pragma once // 只被包含（编译）一次，因此不需要宏定义来避免重定义错误

#include <string>

// 从文件读取 BAL dataset
class BALProblem {
    public:
    // load bal data from text file
    explicit BALProblem( const std::string &filename, bool use_quaternions = false);
    // explicit 只能直接初始化，不能隐式转换。 隐式转换由编译器或解释器自动完成，无需程序员显式指定。

    ~BALProblem() { // 析构函数：在对象销毁时释放动态分配的内存，可防止内存泄漏。
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    // save results to text file
    void WriteToFile( const std::string &filename) const; // 成员函数末尾加 const，表示该成员函数不会修改类的任何成员变量

    // save results to ply pointcloud
    void WriteToPLYFile( const std::string &filename) const;

    void Normalize();

    void Perturb( const double rotation_sigma, 
                const double translation_sigma,
                const double point_sigma);
    
    int camera_block_size() const {return use_quaternions_ ? 10 : 9; } ; // ?:三元条件运算符。这部分函数较简单，就只在头文件定义

    int point_block_size() const { return 3; }

    int num_cameras() const { return num_cameras_; }

    int num_points() const { return num_points_; }

    int num_observations() const { return num_observations_; }

    int num_parameters() const { return num_parameters_; }

    const int *point_index() const { return point_index_; }

    const int *camera_index() const { return camera_index_; }

    const double *observations() const {return observations_; }

    const double *parameters() const {return parameters_; }

    const double *cameras() const { return parameters_; } // 都是返回 parameters_ ？

    const double *points() const { return parameters_ + camera_block_size() * num_cameras_; }

    // camera参数的起始地址
    double *mutable_cameras() { return parameters_; }

    double *mutable_points() { return parameters_ + camera_block_size() * num_cameras_; }

    double *mutable_camera_for_observation( int i ) {
        return mutable_cameras() + camera_index_[i] * camera_block_size();
    }

    double *mutable_point_for_observation( int i ) {
        return mutable_points() + point_index_[i] * point_block_size();
    }

    const double *camera_for_observation( int i ) const {
        return cameras() + camera_index_[i] * camera_block_size();
    }

    const double *point_for_observation( int i ) const {
        return points() + point_index_[i] * point_block_size();
    }

    private:
    void CameraToAngleAxisAndCenter( const double *camera,
                                    double *angle_axis,
                                    double *center) const;
    
    void AngleAxisAndCenterToCamera( const double *angle_axis,
                                    const double *center,
                                    double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    int *point_index_;  // 每个observation对应的 point index
    int *camera_index_; // 每个observation对应的 camera index
    double *observations_;
    double *parameters_;
};
