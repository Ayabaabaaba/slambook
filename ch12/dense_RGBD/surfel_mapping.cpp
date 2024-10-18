#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT;
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

SurfelCloudPtr reconstructSurface( // 函数：从点云中重建表面。返回 SurfelCloudPtr 类型的数据
    const PointCloudPtr &input, float radius, int polynominal_order ) {
    
    pcl::MovingLeastSquares<PointT, SurfelT> mls; // 创建MLS（Moving Least Squares，移动最小二乘法）对象  
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>); // 创建KD树搜索对象，用于快速查找点云中的最近邻 
    // KD树是一种特殊的二叉树，每个节点代表K维空间的一个点
    mls.setSearchMethod(tree); // 设置MLS的搜索方法为KD树  
    mls.setSearchRadius(radius); // 设置搜索半径
    mls.setComputeNormals(true); // 设置是否计算法线 
    mls.setSqrGaussParam(radius * radius); // 设置高斯参数（与搜索半径的平方成正比） 
    // mls.setPolynomialFit( polynominal_order > 1 ); // FIt()为旧版本的函数，表示开启多项式拟合的条件。现在一般直接使用.setPolynomialOrder ？
    mls.setPolynomialOrder( polynominal_order ); // 设置多项式阶数
    mls.setInputCloud( input ); // 设置输入点云  
    SurfelCloudPtr output( new SurfelCloud ); // 创建输出表面元素云  
    mls.process( *output ); // 处理输入点云，生成输出表面元素云 
    return ( output );
}

pcl::PolygonMeshPtr triangulateMesh( const SurfelCloudPtr &surfels) { // 函数：对表面元素进行三角剖分  
    // Create search tree*
    pcl::search::KdTree<SurfelT>::Ptr tree( new pcl::search::KdTree<SurfelT> ); // 创建KD树搜索对象，用于快速查找表面元素云中的最近邻
    tree->setInputCloud(surfels);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<SurfelT> gp3; // 创建贪婪投影三角剖分（Greedy Projection Triangulation）对象 
    pcl::PolygonMeshPtr triangles( new pcl::PolygonMesh ); // 创建多边形网格对象，用于存储三角剖分结果  

    // 设置贪婪投影三角剖分的参数  
    gp3.setSearchRadius(0.05); // Set the maximum distance between connected points (maximum edge length). 设置最大连接点距离（最大边长）
    // Set typical values for the parameters
    gp3.setMu(2.5); // 设置法线之间的角度权重  
    gp3.setMaximumNearestNeighbors(100); // 设置每个点考虑的最大最近邻数量 
    gp3.setMaximumSurfaceAngle( M_PI / 4 );  // 设置最大表面角度（45度）  
    gp3.setMinimumAngle(M_PI / 18 ); // 设置最小角度（10度）
    gp3.setMaximumAngle(2 * M_PI / 3); // 设置最大角度（120度） 
    gp3.setNormalConsistency(true); // 设置法线一致性检查
    // 设置输入表面元素云和搜索方法，进行三角剖分
    gp3.setInputCloud(surfels);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*triangles);

    return triangles;
}

int main(int argc, char **argv) {
    // Load the points. 加载点云  
    PointCloudPtr cloud(new PointCloud); 
    if ( argc == 0 || pcl::io::loadPCDFile(argv[1], *cloud) ) {
        cout << "failed to load point cloud!" ;
        return 1;
    }
    cout << "point cloud loaded, points: " << cloud->points.size() << endl;

    // Compute surface elements. 计算表面元素（法线和平滑表面）(计算点云的法线)
    cout << "computing normals ... " << endl;
    double mls_radius = 0.05, polynominal_order = 2;  // 设置MLS的参数 
    auto surfels = reconstructSurface( cloud, mls_radius, polynominal_order );  // 调用重建表面函数

    // Compute a greedy surface triangulation. 对表面元素进行三角剖分  (从法线计算网格)
    cout << "computing mesh ... " << endl;
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels); // 调用三角剖分函数 

    // 可视化三角剖分结果  
    cout << "display mesh ... " << endl;
    pcl::visualization::PCLVisualizer vis; // 创建PCL可视化对象
    vis.addPolylineFromPolygonMesh(*mesh, "mesh frame"); // 添加多边形网格的边框
    vis.addPolygonMesh(*mesh, "mesh"); // 添加多边形网格 
    vis.resetCamera(); // 重置相机视角
    vis.spin(); // 进入可视化循环
    // 直接关掉窗口会报错，需要额外程序处理窗口关闭事件。PCL的可视化器可能不提供直接的回调机制。
}