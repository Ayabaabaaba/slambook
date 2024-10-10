#include <iostream>
#include <fstream>
#include <string>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Ussage: pose_graph_g2o_SE3 sphere.g2o " << endl;
        return 1;
    }
    ifstream fin(argv[1]);
    if (!fin) {
        cout << "file " << argv[1] << " does not exist." << endl;
        return 1;
    }

    // 设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> BlockSolverType;
    typedef g2o::LinearSolverEigen<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg( // 梯度下降选用LM方法
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>())   );
    g2o::SparseOptimizer optimizer; // 图模型
    optimizer.setAlgorithm(solver); // 设置求解器
    optimizer.setVerbose(true); // 打开调试输出

    int vertexCnt = 0, edgeCnt = 0; // 顶点和边的数量
    while (!fin.eof()) { // !fin.eof() 检查文件流fin是否到达末尾
        string name;
        fin >> name;
        if (name == "VERTEX_SE3:QUAT") { // 读取.g2o文件的节点
            // SE3 顶点
            g2o::VertexSE3 *v = new g2o::VertexSE3(); // 定义新的顶点变量
            int index = 0;
            fin >> index;
            v->setId(index); // 节点ID
            v->read(fin); // 读取数据。SE3类型默认7个，旋转默认按四元数
            optimizer.addVertex(v); // 增加节点到优化器
            vertexCnt++;
            if (index == 0)    v->setFixed(true); // setFixed() 将顶点设置为固定（或锚点），优化过程中该顶点不变
        } else if (name == "EDGE_SE3:QUAT") { // 读取.g2o文件的边
            // SE3-SE3 边
            g2o::EdgeSE3 *e = new g2o::EdgeSE3(); // 定义新的边变量。g2o::EdgeSE3表示两个顶点间的相对变换关系。
            int idx1, idx2; // 关联的两个顶点
            fin >> idx1 >> idx2;
            e->setId(edgeCnt++); // 边ID
            e->setVertex(0, optimizer.vertices()[idx1]); // 边关联节点。在EdgeSE3中，0通常表示起点，1表示终点。
            e->setVertex(1, optimizer.vertices()[idx2]);
            e->read(fin); // 读取数据。默认为7个相对位姿和信息矩阵。EdgeSE3相应的信息矩阵为6*6
            optimizer.addEdge(e); // 增加边到优化器
        }
        if (!fin.good())   break; // !fin.good() 检测文件流fin是否读取良好。如果读到末尾、读取错误、流设置为失败，则fin.good()可能返回false
    }

    cout << "read total " << vertexCnt << " vertices, " << edgeCnt << " edges." << endl;
    
    cout << "optimizing ..." << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30); // 最大迭代次数30次

    cout << "saving optimization results ..." << endl;
    optimizer.save("./build/result.g2o");

    return 0;
}
