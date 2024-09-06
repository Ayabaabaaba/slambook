cmake_minimum_required( VERSION 3.22 )
project( geometry )

# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )
add_executable(eigenGeometry eigenGeometry.cpp) # 生成可执行文件