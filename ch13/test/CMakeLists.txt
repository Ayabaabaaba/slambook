# 定义一个变量 TEST_SOURCES，用于包含测试源文件名（或标识符）的列表
SET(TEST_SOURCES test_triangulation)

FOREACH (test_src ${TEST_SOURCES}) # 开始循环。遍历 TEST_SOURCES 列表中的变量，每次迭代列表中的一个元素（test_triangulation）会被赋值给变量 test_src
    # 添加一个可执行程序： add_executable( 可执行程序名 源代码文件.cpp )
    ADD_EXECUTABLE(${test_src} ${test_src}.cpp)
    # 链接可执行程序与库： target_link_libraries( 可执行程序名 所有第三方库 自定义库列表 )
    TARGET_LINK_LIBRARIES(${test_src} ${THIRD_PARTY_LIBS} myslam)
    # 向CMake的测试套件中添加一个测试。 两个参数分别为：测试的名称、要执行的命令
    ADD_TEST(${test_src} ${test_src})
ENDFOREACH (test_src) # 结束循环
