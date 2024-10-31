#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

DEFINE_string(config_file, "./config/default.yaml", "config file path"); 
// 使用gflags定义了一个命令行参数。参数名；默认值(命令行没有提供时使用，可在命令行指定不同路径覆盖)；参数说明

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true); // 用于解析命令行参数。
    // gflags 库会找命令行中提供的所有参数，并尝试将它们与先前定义的参数（如 config_file）匹配。如果找到了匹配项，它会更新全局变量 FLAGS_config_file 的值，该变量是由 DEFINE_string 宏自动生成的，用于存储 config_file 参数的值。
    // 定义参数（DEFINE_string）是解析参数（google::ParseCommandLineFlags）的前提。如果没有先定义参数，解析器就不知道该识别哪些参数。
    // &argc和&argv允许gflags库修改这两个参数，以便可以去掉已经被解析的参数。
    // true 指示gflags库在解析完参数后，是否应该打印出未识别的参数。设置为true，则打印。

    myslam::VisualOdometry::Ptr vo( new myslam::VisualOdometry(FLAGS_config_file)  );
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}