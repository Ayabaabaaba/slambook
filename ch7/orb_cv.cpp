#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv){
    if (argc != 3 ){
        cout << "usage: feature_extraction img1 img2" << endl;
        return 1;
    }

    // 读取图像
    Mat img_1 = imread(argv[1], IMREAD_COLOR); // cv::IMREAD_COLOR 以参色模式读取图像的标志之一。正确格式为imread(argv[1], IMREAD_COLOR)
    Mat img_2 = imread(argv[2], IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr); // assert() 是C++标准库的一个宏。若断言的条件为假(0)，则程序终止并显示一条错误信息。

    // 初始化（常用的格式）
    std::vector<KeyPoint> keypoints_1, keypoints_2; // 声明 std::vector<KeyPoint> 类型的变量，用于存储检测到的关键点。Keypoint 是 OpenCV 中用于表示关键点的结构体，包含位置、大小、方向等信息。
    Mat descriptors_1, descriptors_2; // 声明 Mat 类型的变量。Mat 是 OpenCV 中用于存储描述符的结构。
    // 智能指针：一种可自动动态分配内存的指针，内存释放时可避免内存泄漏的风险。
    Ptr<FeatureDetector> detector = ORB::create(); // Ptr 是 OpenCV 中用于智能指针的模板类。Ptr<FeatureDetector> 创建一个指向 FeatureDetector特征点检测器 的智能指针。ORB::create() 将其初始化为ORB特征检测器（工厂方法）。
    Ptr<DescriptorExtractor> descriptor = ORB::create(); // Ptr<DescriptorExtractor> 创建一个指向 DescriptorExtractor描述子提取器 的智能指针。ORB特征检测器可同时实现特征检测和描述符提取的功能。
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // DescriptorMatcher描述子匹配器。 
    // DescriptorMatcher::create("BruteForce-Hamming") 初始化为使用Hamming距离的暴力匹配器。create 方法是工厂方法，用于根据给定字符串，创建匹配器的实例。

    // 第一步：检测 Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); 
    detector->detect(img_1, keypoints_1); // 根据前文 detector 为指向 FeatureDetector特征点检测器 的智能指针。
    // detector 的 detect()，参数分别为：要检测的图像、存储特征点位置的变量。
    detector->detect(img_2, keypoints_2); 
    
    // 第二步：根据角点位置，计算 BRIEF描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1); // descriptor 的 compute() 计算已经检测到的特征点(角点)的BRIEF描述子。输入参数：输入图像、特征点集合、用于存储描述子的矩阵。
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT); // drawKeypoints() 在原始图像上，绘制特征点集合中的特征点，并将结果存储在新的图像。
    // drawKeypoints 输入参数：原始图像、特征点集合、存储结果的图像、指定特征点绘制的颜色、绘制细节控制。
    // Scalar::all(-1) 使用默认颜色(通常为亮色)。也可指定具体颜色，比如 Scalar(255, 0, 0)表示蓝色。
    // DrawMatchesFlags::DEFAULT 使用默认设置的绘制标志。
    imshow("ORB features", outimg1); // 此时只有单幅图的特征点，还未进行相邻帧的匹配。

    // 第三步：根据角点位置计算 BRIEF 描述子
    vector<DMatch> matches; // DMatch 一个结构体，包含两个描述子间的匹配信息。比如两张相邻帧的描述子的索引，以及匹配的质量。
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches); // match() 输入两组描述子，并将找到的匹配项存储在 matches 向量。matcher 是前面定义的描述子匹配器的智能指针。
    // 即使两组描述子的数量不同，每个 descriptors_2 的描述子都会被处理，并尝试在 descriptors_1 中找到匹配项。若 descriptors_1 的描述子较少，在 descriptors 的部分描述子可能无法找到匹配项。
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

    // 第四步：匹配点对筛选（去除误匹配）
    // 计算最小距离和最大距离
    // 迭代器：一种访问容器中元素的方法（不需要了解容器的内部结构）。相当于容器的指针，但更灵活，可以“迭代”或遍历容器中的元素。
    // lambda函数：匿名函数在C++的表达。匿名函数没有函数名，用于定义简单的、一次性的函数对象。
    // lambda函数的格式为： [capture](parameters) mutable -> return_type {//函数体}
    // capture 指定外部变量的捕获方式。parameters 参数列表。 mutable 可选的，指定lambda体内变量可修改(一般体内变量是只读的)。-> return_type 指定返回类型(默认可自动推断类型)。
    auto min_max = minmax_element(matches.begin(), matches.end(), // minmax_element() C++标准库的算法，返回一对迭代器。
        [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; } ); // 输入参数：范围的开始迭代器、范围的结束迭代器、可选的比较函数(用于确定两元素间的顺序)
        // lambda函数 [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; } 输入两个DMatch类性的常量引用，返回一个bool值，指示m1的distance是否小于m2的distance
        // m1和m2：当前正在比较的两个DMatch对象的引用。每当minmax_element算法需要比较两个元素时，会通过迭代器获取这两个元素的引用。
        // minmax_element 函数返回一个 std::pair<Iterator, Iterator> 类型的值。Iterator 是输入范围的迭代器类型，这个 pair 的 .first 成员是一个迭代器（指向范围内的最小元素），.second 是另一个迭代器(指向范围内的最大元素)
    double min_dist = min_max.first->distance; 
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误。（经验方法，仍需要更可行的误匹配筛选算法）
    // 有时最小距离会非常小,设置一个经验值30作为下限。
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++){ // 这里 descriptors_1.rows 改为 matches.size() 可能更好，因为 matches 中的匹配项可能少于 descriptors_1 的描述子数量
        if (matches[i].distance <= max(2 * min_dist, 30.0)){
            good_matches.push_back(matches[i]);
        }
    }

    // 第五步：绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    // drawMatches() 是OpenCV中，在两张图像上绘制匹配的特征点对的函数。
    // drawMatches输入参数：img_1、img_2为要绘制匹配点的两张图像，keypoints_1、keypoints_2两张图的特征点集合，matches 为DMatch对象的向量(含有两组关键点的匹配信息)，img_match带有绘制匹配结果的输出图像
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey(0);

    return 0;
}