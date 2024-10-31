# 创建一个库目标 add_library(<name> [<TYPE>] [SOURCE_FILES])
# <TYPE> 库类型：STATIC（静态库，默认）、SHARED（共享库）、MODULE（动态加载的模块，主要用于插件）。
add_library(myslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        visual_odometry.cpp
        dataset.cpp)

# 链接库目标(.cpp成员定义)与库:  target_link_libraries( 库目标 库名 )
target_link_libraries(myslam ${THIRD_PARTY_LIBS})