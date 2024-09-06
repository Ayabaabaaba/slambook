cmake_minimum_required(VERSION 3.22) 
project(plotTraj) 

set(CMAKE_CXX_STANDARD 11)

# 如果检索不到eigen3库，可能需要包含Eigen库的头文件
# include_directories("/usr/include/eigen3")

# 
include_directories("/usr/include/eigen3")
add_executable(coordinateTransform coordinateTransform.cpp)

# 
find_package(Pangolin REQUIRED) # 查找Pangolin库的目录。
# REQUIRED 如果找不到会报错并停止。
# 如果找到了，头文件目录为 Pangolin_INCLUDE_DIRS ，库文件目录为 Pangolin_LIBRARIES
include_directories(${Pangolin_INCLUDE_DIRS}) # 包含库的头文件目录
add_executable(plotTrajectory plotTrajectory.cpp)  # 添加可执行文件的编译
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES}) # 链接库文件和可执行程序

# 使用Pangolin库时，可能还需要手动链接OpenGL库
# find_package(OpenGL REQUIRED)
# include_directories(${OPENGL_INCLUDE_DIRS})
# target_link_libraries(plotTrajectory ${OPENGL_LIBRARIES})

# 同时也可能需要手动链接GLEW库
# find_package(GLEW REQUIRED)
# include_directories(${GLEW_INCLUDE_DIRS})
# target_link_libraries(plotTrajectory ${GLEW_LIBRARIES})
 