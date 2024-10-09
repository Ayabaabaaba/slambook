#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    if (argc != 2 ) {
        cout << "usage: bundle_adjustment_ceres_bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]); // 构造函数，读取BAL数据
    bal_problem.Normalize(); // 场景归一化
    bal_problem.Perturb(0.1, 0.5, 0.5);  // 往路标点和相机的旋转、平移添加标准正态分布扰动
    bal_problem.WriteToPLYFile("./build/innitial.ply"); // 将初始的观测数据、相机参数、路标3D点（含扰动）写入PLY文件
    SolveBA(bal_problem); // 求解BA问题
    bal_problem.WriteToPLYFile("./build/final.ply"); // 将优化求解后的参数写入另一个PLY文件（便于MeshLab查看）

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size(); // 坐标维度3
    const int camera_block_size = bal_problem.camera_block_size(); // 相机参数维度9或10
    double *points = bal_problem.mutable_points(); // BAL数据中路标3D点的起始位置
    double *cameras = bal_problem.mutable_cameras(); // BAL数据中相机参数的起始位置

    // Observations is 2 * num_observations long array observations [u_1, u_2, ..., u_n],
    // where each u_i is two dimensional, the x and y position of the observation.
    const double *observations = bal_problem.observations(); // BAL数据中观测数据的起始位置

    ceres::Problem problem; // 创建一个ceres::Problem对象。用于构建最小二乘问题
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function; // 定义ceres::CostFunction对象
        
        // Each Residual block takes a point and a camera as input and outputs a 2 dimensional Residual.
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]); // 自定义的残差类，创建投影残差，并返回ceres::AutoDiffCostFunction对象。后续加入problem.AddResidualBlock()中。

        // If enabled use Huber's loss function. 定义核函数。
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0); // Huber核，阈值参数为1.0

        // Each observation corresponds to a pair of a camera and a point 
        // which are identified by camera_inde()[i] and point_index()[i] respectively. 待估计参数的初始值。
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        problem.AddResidualBlock( cost_function,  loss_function,  camera,  point); // 残差块可以一直按顺序添加
    }

    // show some information here ...
    std::cout << "bal problem file loaded ... " << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() 
            << " cameras and " << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options; // Ceres求解的选项
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; // 求解方法设定为 SPARSE_SCHUR ，先对路标部分进行Schur边缘化，加速求解
    options.minimizer_progress_to_stdout = true; // 向cout输出每次迭代的结果
    ceres::Solver::Summary summary; // 完成优化过程后返回的结构体，包含优化过程详细信息
    ceres::Solve(options, &problem, &summary); // ceres库开始求解
    std::cout << summary.FullReport() << "\n"; // 输出优化的完整结果
}
