# 添加一个可执行程序： add_executable( 可执行程序名 源代码文件.cpp )
add_executable(run_kitti_stereo run_kitti_stereo.cpp)
# 链接可执行程序与库:  target_link_libraries( 可执行程序名 库名 )
target_link_libraries(run_kitti_stereo myslam ${THIRD_PARTY_LIBS})