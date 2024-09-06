#include <pangolin/pangolin.h>
#include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Geometry>
#include <unistd.h>

using namespace std;
using namespace Eigen;

string trajectory_file = "./examples/trajectory.txt"; // 存放轨迹数据的文件

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>); 
// 函数声明。函数的结构体定义在main函数下方
// void 函数没有返回值
// vector<>中，第一个参数Isometry3d用于指定vector中存储的数据类型
// Eigen::aligned_allocator是专门用于Eigen类型的分配器，用于动态分配和释放内存，确保分配的内存满足Isometry3d类型的对齐方式

int main(int argc, char **argv){
    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses; 
    // 定义了一个变量（尚未赋值）
    
    ifstream fin(trajectory_file); 
    // 以只读模式打开指定文件
    // 若打开失败， !fin == true ； 若打开成功 !fin == false
    // .eof()用于判断是否成功读取到文件末尾。
    // 若打开成功，但未读取到文件末尾，fin.eof() == false ；若读取到末尾，fin.eof() == true

    if(!fin){
        cout << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }

    while(!fin.eof()){ 
        double time, tx, ty, tz, qx, qy, qz, qw; 
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw; // 从fin关联的文件中顺序读取8个值，并分别赋给8个变量
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
        Twr.pretranslate(Vector3d(tx, ty, tz)); // .pretranslate()向欧氏变换矩阵添加平移向量
        poses.push_back(Twr); // .push_back()用于将变量(或指针，不限定类型？)存储到poses容器的末尾
    }
    cout << "read total " << poses.size() << " pose entries" << endl; // 姿态的数据组数

    DrawTrajectory(poses); // 通过函数绘制轨迹
    return 0;
}

/************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses){ 
    
    // 初始化Pangolin
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768); // 创建一个名为 "Trajectory Viewer" 的窗口，并设置其大小为 1024x768 像素。
    glEnable(GL_DEPTH_TEST); // 启用深度测试，以便正确处理深度信息
    glEnable(GL_BLEND); // 启用混合（blending），以便正确处理透明效果
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam( // 用于定义相机的投影矩阵和模型视图矩阵。
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000), // 设置相机的位置和观察方向。
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay() // d_cam 是一个视图对象，与 s_cam 关联，并设置了其在窗口中的位置和大小。
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam)); // 绑定了一个 Handler3D，允许用户通过鼠标和键盘与相机交互

    while (pangolin::ShouldQuit() == false) { // 一个标准的 OpenGL 渲染循环，直到用户请求退出（例如，关闭窗口）。
        // pangolin::ShouldQuit() 当用户请求退出（如按Esc、关闭窗口等）时，返回true

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清除屏幕和深度缓冲区
        d_cam.Activate(s_cam); // 激活与d_cam关联的s_cam的渲染状态
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // 设置OpenGL的清除颜色为纯白（即屏幕颜色）
        glLineWidth(2); // 设置OpenGL线条绘制的线宽

        for (size_t i = 0; i < poses.size(); i++){
            // 画每个位姿的三个坐标轴
            Vector3d Ow = poses[i].translation(); // .translation()用于获取平移向量的部分
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
            glBegin(GL_LINES); // 指定接下来的顶点用于绘制线段
            glColor3f(1.0, 0.0, 0.0); // 分别为RGB强度，最低0.0,最高1.0。这里为红色
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0); // 这里为绿色
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0); // 这里为蓝色
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd(); // 与glBegin()相对应
        }
        // 画出姿态之间的连线
        for (size_t i = 0; i < poses.size() - 1; i++){
            glColor3f(0.0, 0.0, 0.0); // 这里为黑色
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        pangolin::FinishFrame(); // 完成所有OpenGL调用后，交换缓冲区并检查事件
        usleep(5000); // sleep 5 ms
    }
}