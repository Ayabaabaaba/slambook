#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main( int argc, char** argv ){
    string dataset_dir = argv[1]; // 从命令行参数获取数据集目录的路径  
    ifstream fin ( dataset_dir + "/associate.txt"); // 打开包含图像和时间戳关联信息的文件
    if ( !fin ) {
        cout<<"please generate the associate file called associate.txt!"<<endl;
        return 1;
    }

    // 定义存储图像文件名、时间戳的向量
    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;
    // 从文件中读取图像和时间戳信息，并存储到相应的向量中  
    while ( !fin.eof() ) {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back( atof(rgb_time.c_str()) ); // atof() 为 ASCII to float 缩写，表示将字符串转换为浮点数。
        depth_times.push_back( atof(depth_time.c_str()) );
        rgb_files.push_back( dataset_dir + "/" + rgb_file );
        depth_files.push_back( dataset_dir + "/" + depth_file );

        if ( fin.good() == false )    break;     
    }
    fin.close();

    // 提取ORB特征
    cout << "generating features ... " << endl;
    vector<Mat> descriptors;
    Ptr<Feature2D> detector = ORB::create();
    int index = 1;
    for ( string rgb_file:rgb_files ) { // rgb_file:rgb_files 遍历rgb_files的元素，每次将元素赋予rgb_file
        Mat image = imread( rgb_file );
        vector<KeyPoint> keypoints;
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
        cout << "extracting features from image " << index++ << endl;
    }
    cout << "extract total " << descriptors.size() * 500 << " features. " << endl;

    // create vocabulary
    cout << "creating vocabulary, please wait ... " << endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors ); // 通过描述符集合，创建字典
    cout << "vocabulary info: " << vocab << endl; // 输出字典信息
    vocab.save( "./build/vocab_larger.yml.gz "); // 保存字典到指定文件
    cout << "done" << endl;

    return 0;
}