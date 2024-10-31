cmake_minimum_required(VERSION 2.8) # 指定CMake的最低版本要求为2.8
project(myslam) # 定义项目名称为myslam。

set(CMAKE_BUILD_TYPE Release) # 设置默认的构建类型为Release，这意味着优化将被开启，调试信息将被最小化。

set(CMAKE_CXX_FLAGS "-std=c++17 -Wall") # 为所有构建类型设置C++编译器标志，使用C++17标准和开启所有警告
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++17 -O3 -fopenmp -pthread") # 为Release构建类型设置特定的C++编译器标志，包括C++17标准、高级优化、开启OpenMP并行支持和线程支持。

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules) # cmake_modules目录的自定义cmake模块，通常为FindCSparse.cmake、FindG2O.cmake、FindGlog.cmake等寻找库的
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) # 设置可执行程序的输出路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) # 设置库文件的输出路径。作用？
# # 设置cmake中间变量的输出路径
# set(CMAKE_OBJECT_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)  
# set(CMAKE_OBJECT_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/build)  
# set(CMAKE_OBJECT_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/build)

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV 4.10 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# 终端运行 pkg-config --modversion opencv4 查看 OpenCV 库版本

# pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
# REQUIRED 如果找不到会报错并停止。
# 如果找到了，头文件目录为 Pangolin_INCLUDE_DIRS ，库文件目录为 Pangolin_LIBRARIES

# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# G2O
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# glog
find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

# gtest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# gflags
find_package(GFlags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# 定义一个变量THIRD_PARTY_LIBS，包含所有第三方库的链接器标志
# glut库需要额外安装 sudo apt-get install freeglut3-dev 不过这里没用到glut库
set(THIRD_PARTY_LIBS 
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${Pangolin_LIBRARIES} GL GLU GLEW # glut
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSPARSE_LIBRARY}
        )

# 启用测试功能，这通常与add_test命令一起使用，用于添加测试
enable_testing() 

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/include) # 此处无CMAkeLists.txt
add_subdirectory(src) # 此处（有.cpp）需要CMakeLists.txt （自定义库的成员定义。创建库目标列表、并链接所有第三方库）
add_subdirectory(test) # 此处（有.cpp）需要CMakeLists.txt （测试文件。）
add_subdirectory(app) # 此处（有.cpp）需要CMakeLists.txt (main函数。添加可执行程序，并链接所有第三方库)
