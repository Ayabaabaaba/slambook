# 声明要求的cmake最低版本
cmake_minimum_required( VERSION 3.22 )

# 声明一个cmake工程
project( HelloSLAM )

# 添加一个可执行程序
# 语法：add_executable( 程序名 源代码文件 )
add_executable( helloSLAM helloSLAM.cpp )

# 生成静态库
add_library( hello libHelloSLAM.cpp)

# 生成共享库
add_library(hello_shared SHARED libHelloSLAM.cpp)

# 生成新写的程序，并链接头文件
add_executable( useHello useHello.cpp)
target_link_libraries( useHello hello_shared )
