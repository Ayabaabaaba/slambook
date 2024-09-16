project(imageBasics)

# 添加C++ 11 标准支持
set( CMAKE_CXX_FLAGS "-std=c++11")

# OpenCV库
find_package( OpenCV REQUIRED) # 寻找OpenCV库
include_directories( ${OpenCV_INCLUDE_DIRS} ) # 添加OpenCV库的头文件

add_executable(imageBasics imageBasics.cpp) # 添加可执行程序
target_link_libraries(imageBasics ${OpenCV_LIBS}) # 链接OpenCV库

# add_executable(undistortImage undistortImage.cpp)
# target_link_libraries(undistortImage ${OpenCV_LIBS}) # 链接OpenCV库