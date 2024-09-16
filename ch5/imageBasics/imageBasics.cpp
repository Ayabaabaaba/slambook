#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv){
    // 读取argv[1]指定的图像，即在命令行中运行可执行程序时，程序名后输入的其他命令
    cv::Mat image;
    image = cv::imread(argv[1]); // cv::imread读取指定路径下的图像

    // 判断图像是否读取正确
    if (image.data == nullptr){ // 数据不存在，可能路径错误，也可能文件不存在
        cerr << "文件" << argv[1] << "不存在." << endl; 
        // cout 标准输出流，是缓冲的，会暂时存储在内存中，知道缓冲区满或显示刷新(如endl)才会实际输出。
        // cerr 标准错误输出流，是非缓冲的，写入的信息会立即发送到输出设备。
        // cout 的输出可以重定向到文件，cerr的输出仍显示在控制台。
        return 0;
    }

    // 顺利读取文件时，先输出基本信息
    cout << "图像宽为" << image.cols << ",高为" << image.rows << ",通道数为" << image.channels() << endl;
    cv::imshow("image", image); // cv::imshow显示图像
    cv::waitKey(0); // 暂停，待输入一个按键(要在打开的图片窗口中输入)

    // 判断image类型
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) { 
        // CV_8UC1表示8位无符号单通道图像，即灰度图像。
        // CV_8UC3表示8位无符号三通道图像，即彩色图像（在OpenCV中通常为BGR格式）。
        cout << "请输入一张彩色图或灰度图." << endl; // 若类型不符合要求，则输出并终止
        return 0;
    }

    // 遍历图像，以下遍历方式也可使用于随机像素访问（即针对任一像素，而非逐行逐列）
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); // std::chrono 用于给算法计时
    for (size_t y = 0; y < image.rows; y++){ // 按数逐行(即图像高度)遍历
        
        unsigned char *row_ptr = image.ptr<unsigned char>(y); // row_ptr是第y行的头指针。cv::Mat::ptr用于获取图像的行指针。
        // unsigned char (无符号字符)类型常用于表示灰度图像；彩色图像的三个通道中，每个通道同样以unsigned char类型存储

        for (size_t x = 0; x < image.cols; x++){
            unsigned char *data_ptr = &row_ptr[x * image.channels()]; // data_ptr 指向待访问的像素数据
            // * 为解引用操作符，用于获取指针所指向的值。& 为取地址操作符，用于获取变量的内存地址。
            // image.channels()函数返回图像的通道数，灰度图为1,彩色图为3.
            // 对于灰度图，应该访问 row_ptr[x]；对于彩色图，应该访问 row_ptr[x * image.channels() + c]，其中 c 是通道索引（0为B，1为G，2为R）？

            // 访问该像素的每个通道（可用于输出查看）
            for (int c = 0; c != image.channels(); c++){
                unsigned char data = data_ptr[c]; // 数据data为I(x,y)第c个通道的值。
                // data_ptr 是一个指针，指向像素I(x,y)的第一个通道。访问第一个通道时需使用 *data_ptr
                // data_ptr[c] 由c提供偏移量，并且本身包含解指针引用的功能，不需要*。同时，unsigned char类型里，每个通道的指针相差1(1个字节)。
            }

        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now(); // 算法结束时间
    // chrono::steady_clock 为C++11引入的时钟类型，提供单调递增的时间点，不受系统时间调整的影响。
    // chrono::time_point 为时间点类型， chrono::steady_clock::time_point 是与 chrono::steady_clock 相关联的时间点类型。
    chrono::duration<double> time_used = chrono::duration_cast < chrono::duration < double >> (t2 - t1); 
    // chrono::duration 表示时间间隔，可修改单位和精度类型，默认为秒。<double>是为了获取更高精度，也可改为其他，如int（会限制精度）。
    // chrono::duration_cast 一个转换时间间隔类型的模板函数，将 t2 - t1 的类型转换为 chrono::duration < double > 类型。
    cout << "遍历图像用时：" << time_used.count() << " 秒。" << endl;
    // .count() 函数是 chrono::duration 类的一个成员函数，返回时间间隔的数值部分。

    // 对于 cv::Mat ，直接赋值和拷贝数据的区别
    // 直接赋值不会拷贝数据，修改 image_another 会导致 image 发生变化
    cv::Mat image_another = image; // 直接赋值
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0); // 修改左上角100*100的块置零
    cv::imshow("image",image); // 原image也会发生变化
    cv::waitKey(0); // 暂停程序,等待一个按键输入
    // 使用 .clone() 才能拷贝数据
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    // 图像的其他基本操作：剪切、旋转、缩放等。参看OpenCV官方文档

    cv::destroyAllWindows(); // OpenCV 结束语，关闭所有窗口
    return 0;
}