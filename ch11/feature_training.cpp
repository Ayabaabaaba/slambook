#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/*----训练./data/目录下的十张图像的字典-----*/
int main( int argc, char** argv ) {
    // read the image
    cout << "reading images ... " << endl;
    vector<Mat> images;
    for (int i = 0; i < 10; i++) {
        string path = "./data/" + to_string(i+1) + ".png"; // 按文件名顺序依次读取。mark
        images.push_back( imread(path) );
    }

    // detect ORB features
    cout << "detecting ORB features ... " << endl;
    Ptr< Feature2D > detector = ORB::create(); 
    // Ptr<Feature2D> 创建一个指向 Feature2D 的智能指针。ORB::create() 将其初始化为ORB特征检测器。后面检测的为ORB特征。
    vector<Mat> descriptors;
    for ( Mat& image:images ) {
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor ); 
        // detectAndCompute() 检测关键点和计算描述符。输入参数：输入图像，可选的输入掩码(可指定处理的图像区域，为空则整个图像)，keypoints 输出的关键点，descriptor 输出的描述符。
        descriptors.push_back( descriptor );
    }

    // create vocabulary
    cout << "creating vocabulary ... " << endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors ); // DBoW3::Vocabulary 类的 .create()，根据提供的特征描述符集合创建词汇表。不关注单词的位置，故不需要keypoints
    // 默认使用聚类算法提取单词，并非所有描述符都会转换为单词。
    cout << "vocabulary info: " << vocab << endl << endl;
    vocab.save( "./build/vocabulary.yml.gz" );
    cout << endl << "done" << endl;

    return 0;
}