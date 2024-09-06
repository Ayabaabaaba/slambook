cmake_minimum_required(VERSION 3.22) # 指定cmake最低版本要求为3.22
project(useEigen) # 将该cmake项目命名为useEigen

set(CMAKE_BUILD_TYPE "Release") # cmake的构建类型，有"Release"和"Debug"两种。"Release"禁用调试，使编译更快但更难调试
set(CMAKE_CXX_FLAGS "-O3") # 较为粗略的粒度控制，用以更快生成执行代码

# 添加Eigen头文件
include_directories("/usr/include/eigen3") # 添加库的头文件
add_executable(eigenMatrix eigenMatrix.cpp) # 将源文件.cpp生成可执行文件(无后缀)