
/**************************************************
Copyright(c) 2018-2018 fellen All rights reserved. 
Author: fellen
Date:2018-08-20 
Description:1、Compute starting translation and rotation based on MomentOfInertiaEstimation descriptor
            2、LM-ICP Alignment                                                                                                                                          
**************************************************/

//ROS头文件
#include <ros/ros.h>
//ROS数据格式与PCL数据格式转换
#include <pcl_conversions/pcl_conversions.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
//boost指针管理
#include <boost/make_shared.hpp> 
//点/点云
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
//pcd文件输入/输出
#include <pcl/io/pcd_io.h>
//滤波
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
//特征
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
//配准
#include <pcl/registration/icp.h> 
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
//可视化
#include <pcl/visualization/pcl_visualizer.h>
//opencv2
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//image transport
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
//算法时间测试
#include <pcl/common/time.h>
//字符串
#include <string>
//tf坐标系变换
#include <tf/transform_broadcaster.h>

//命名空间
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//定义类型的别名
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

//**************全局变量**************//
//ros发布对象
ros::Publisher pub;
ros::Subscriber depth_sub;
ros::Subscriber MaskRCNN_sub;

ros::Publisher pcl_pub; 

//可视化对象
pcl::visualization::PCLVisualizer *p;
//左视区和右视区，可视化窗口分成左右两部分
int vp_1, vp_2;
//ROS图像转OpenCV变量
cv_bridge::CvImagePtr cv_ptr, mask_ptr;
//调试标志
bool DEBUG_WITH_OTHERS = true;
bool DEBUG_VISUALIZER = false;

//定义结构体，用于处理点云
struct PCD
{
  PointCloud::Ptr cloud; //点云指针
  std::string f_name; //文件名
  PCD() : cloud (new PointCloud) {}; //构造函数初始化
};

// 定义新的点表达方式< x, y, z, curvature > 坐标+曲率
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT> //继承关系
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
  public:
  MyPointRepresentation ()
  {
    //指定维数
    nr_dimensions_ = 4;
  }
  //重载函数copyToFloatArray，以定义自己的特征向量
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    //< x, y, z, curvature > 坐标xyz和曲率
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

//在窗口的左视区，简单的显示源点云和目标点云
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
  p->removePointCloud ("vp1_source"); //
  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0); //目标点云绿色
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0); //源点云红色
  p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1); //加载点云
  p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);
  PCL_INFO ("Press q to begin the registration.\n"); //在命令行中显示提示信息
  p-> spin();
}

//在窗口的右视区，简单的显示源点云和目标点云
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud ("source"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
  p->removePointCloud ("target");
  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature"); //目标点云彩色句柄
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");
  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature"); //源点云彩色句柄
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");
  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2); //加载点云
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);
  p->spinOnce();
}

// 读取一系列的PCD文件（希望配准的点云文件）
// 参数argc 参数的数量（来自main()）
// 参数argv 参数的列表（来自main()）
// 参数models 点云数据集的结果向量
void loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  std::string extension (".pcd"); //声明并初始化string类型变量extension，表示文件后缀名
  // 通过遍历文件名，读取pcd文件
  for (int i = 1; i < argc; i++) //遍历所有的文件名（略过程序名）
  {
    std::string fname = std::string (argv[i]);
    if (fname.size () <= extension.size ()) //文件名的长度是否符合要求
      continue;

    std::transform (fname.begin (), fname.end (), fname.begin (), (int(*)(int))tolower); //将某操作(小写字母化)应用于指定范围的每个元素
    //检查文件是否是pcd文件
    if (fname.compare (fname.size () - extension.size (), extension.size (), extension) == 0)
    {
      // 读取点云，并保存到models
      PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloud); //读取点云数据
      //去除点云中的NaN点（xyz都是NaN）
      std::vector<int> indices; //保存去除的点的索引
      pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices); //去除点云中的NaN点
      models.push_back (m);
    }
  }
}


pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 transformation2;
void prePairAlign(const PointCloud::Ptr cloud_src,const PointCloud::Ptr cloud_tgt, PointCloud::Ptr transformed_cloud,  bool downsample)
{
  PointCloud::Ptr src (new PointCloud); //创建点云指针
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid; //VoxelGrid 把一个给定的点云，聚集在一个局部的3D网格上,并下采样和滤波点云数据
  if (downsample) //下采样
  {
    grid.setLeafSize (0.007, 0.007, 0.007); //设置体元网格的叶子大小
        //下采样 源点云
    grid.setInputCloud (cloud_src); //设置输入点云
    grid.filter (*src); //下采样和滤波，并存储在src中
        //下采样 目标点云
    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);

    PCL_INFO ("Partial Pointcloud size after sampling is. %d.\n", src->size());
    PCL_INFO ("Model Pointcloud size after sampling is. %d.\n", tgt->size());
  }
  else //不下采样
  {
    src = cloud_src; //直接复制
    tgt = cloud_tgt;
  }
  //******************************OBB包围盒计算*************************************//
	pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(src);
	feature_extractor.compute();

	std::vector <float> moment_of_inertia;
	std::vector <float> eccentricity;
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;

	feature_extractor.getMomentOfInertia(moment_of_inertia);
	feature_extractor.getEccentricity(eccentricity);
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	feature_extractor.getEigenValues(major_value, middle_value, minor_value);
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
	feature_extractor.getMassCenter(mass_center);

    //***********************************可视化重心、包围盒和坐标系******************************************//
  if(true == DEBUG_VISUALIZER)
  {
    //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    // p->setBackgroundColor(0, 0, 0);
    p->addCoordinateSystem(0.2);
    // p->initCameraParameters();
    p->addPointCloud<pcl::PointXYZ>(cloud_src, "prePairAlign source");
    p->addPointCloud<pcl::PointXYZ>(cloud_tgt, "prePairAlign target");
    //p->addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, "AABB");
  }

	Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
	Eigen::Quaternionf quat(rotational_matrix_OBB);
	//p->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");

	pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
  if(true == DEBUG_VISUALIZER)
  {
    p->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
    p->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
    p->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");
  }

	Eigen::Vector3f p1(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
	Eigen::Vector3f p2(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
	Eigen::Vector3f p3(max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
	Eigen::Vector3f p4(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
	Eigen::Vector3f p5(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
	Eigen::Vector3f p6(min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
	Eigen::Vector3f p7(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
	Eigen::Vector3f p8(max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

	p1 = rotational_matrix_OBB * p1 + position;
	p2 = rotational_matrix_OBB * p2 + position;
	p3 = rotational_matrix_OBB * p3 + position;
	p4 = rotational_matrix_OBB * p4 + position;
	p5 = rotational_matrix_OBB * p5 + position;
	p6 = rotational_matrix_OBB * p6 + position;
	p7 = rotational_matrix_OBB * p7 + position;
	p8 = rotational_matrix_OBB * p8 + position;

	pcl::PointXYZ pt1(p1(0), p1(1), p1(2));
	pcl::PointXYZ pt2(p2(0), p2(1), p2(2));
	pcl::PointXYZ pt3(p3(0), p3(1), p3(2));
	pcl::PointXYZ pt4(p4(0), p4(1), p4(2));
	pcl::PointXYZ pt5(p5(0), p5(1), p5(2));
	pcl::PointXYZ pt6(p6(0), p6(1), p6(2));
	pcl::PointXYZ pt7(p7(0), p7(1), p7(2));
	pcl::PointXYZ pt8(p8(0), p8(1), p8(2));

  if(true == DEBUG_VISUALIZER)
  {
    p->addLine(pt1, pt2, 1.0, 0.0, 0.0, "1 edge");
    p->addLine(pt1, pt4, 1.0, 0.0, 0.0, "2 edge");
    p->addLine(pt1, pt5, 1.0, 0.0, 0.0, "3 edge");
    p->addLine(pt5, pt6, 1.0, 0.0, 0.0, "4 edge");
    p->addLine(pt5, pt8, 1.0, 0.0, 0.0, "5 edge");
    p->addLine(pt2, pt6, 1.0, 0.0, 0.0, "6 edge");
    p->addLine(pt6, pt7, 1.0, 0.0, 0.0, "7 edge");
    p->addLine(pt7, pt8, 1.0, 0.0, 0.0, "8 edge");
    p->addLine(pt2, pt3, 1.0, 0.0, 0.0, "9 edge");
    p->addLine(pt4, pt8, 1.0, 0.0, 0.0, "10 edge");
    p->addLine(pt3, pt4, 1.0, 0.0, 0.0, "11 edge");
    p->addLine(pt3, pt7, 1.0, 0.0, 0.0, "12 edge");
  }


	//************************************计算旋转矩阵***********************************//
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>());

	cloud_in->width = 4;
	cloud_in->height = 1;
	cloud_in->is_dense = false;
	cloud_in->resize(cloud_in->width * cloud_in->height);

	cloud_out->width = 4;
	cloud_out->height = 1;
	cloud_out->is_dense = false;
	cloud_out->resize(cloud_out->width * cloud_out->height);

	//输入四个点
	cloud_in->points[0].x = 0;
	cloud_in->points[0].y = 0;
	cloud_in->points[0].z = 0;

	cloud_in->points[1].x = 1;
	cloud_in->points[1].y = 0;
	cloud_in->points[1].z = 0;

	cloud_in->points[2].x = 0;
	cloud_in->points[2].y = 1;
	cloud_in->points[2].z = 0;

	cloud_in->points[3].x = 0;
	cloud_in->points[3].y = 0;
	cloud_in->points[3].z = 1;

	//目标四个点
	cloud_out->points[0].x = center.x;
	cloud_out->points[0].y = center.y;
	cloud_out->points[0].z = center.z;

	cloud_out->points[1].x = x_axis.x;
	cloud_out->points[1].y = x_axis.y;
	cloud_out->points[1].z = x_axis.z;

	cloud_out->points[2].x = y_axis.x;
	cloud_out->points[2].y = y_axis.y;
	cloud_out->points[2].z = y_axis.z;

	cloud_out->points[3].x = z_axis.x;
	cloud_out->points[3].y = z_axis.y;
	cloud_out->points[3].z = z_axis.z;

	//利用SVD方法求解变换矩阵  
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> TESVD;

	TESVD.estimateRigidTransformation(*cloud_in, *cloud_out, transformation2);
	//输出变换矩阵信息  
	std::cout << "The Pre-estimated Rotation and translation matrices are : \n" << std::endl;
	printf("\n");
	printf("    | %6.3f %6.3f %6.3f | \n", transformation2(0, 0), transformation2(0, 1), transformation2(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", transformation2(1, 0), transformation2(1, 1), transformation2(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", transformation2(2, 0), transformation2(2, 1), transformation2(2, 2));
	printf("\n");
	printf("t = < %0.3f, %0.3f, %0.3f >\n", transformation2(0, 3), transformation2(1, 3), transformation2(2, 3));

	//Executing the transformation
	//pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	//You can either apply transform_1 or transform_2; they are the same
  //pcl::transformPointCloud(*cloud_src, *transformed_cloud, transformation2);
  pcl::transformPointCloud(*cloud_tgt, *transformed_cloud, transformation2);
   if(true == DEBUG_VISUALIZER)
  {
    p->addPointCloud<pcl::PointXYZ>(transformed_cloud, "prePairAlign cloud");
  }
  //pcl::io::savePCDFileASCII("transformed_cloud.pcd", *transformed_cloud);	
}

//简单地配准一对点云数据，并返回结果
//参数cloud_src  源点云
//参数cloud_tgt  目标点云
//参数output     输出点云
//参数final_transform 成对变换矩阵
//参数downsample 是否下采样
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  //为了一致性和速度，下采样
  PointCloud::Ptr src (new PointCloud); //创建点云指针
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid; //VoxelGrid 把一个给定的点云，聚集在一个局部的3D网格上,并下采样和滤波点云数据
  if (downsample) //下采样
  {
    grid.setLeafSize (0.007, 0.007, 0.007); //设置体元网格的叶子大小
        //下采样 源点云
    grid.setInputCloud (cloud_src); //设置输入点云
    grid.filter (*src); //下采样和滤波，并存储在src中
        //下采样 目标点云
    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else //不下采样
  {
    src = cloud_src; //直接复制
    tgt = cloud_tgt;
  }

  //计算曲面的法向量和曲率
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals); //创建源点云指针（注意点的类型包含坐标和法向量）
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals); //创建目标点云指针（注意点的类型包含坐标和法向量）
  pcl::NormalEstimation<PointT, PointNormalT> norm_est; //该对象用于计算法向量
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ()); //创建kd树，用于计算法向量的搜索方法
  norm_est.setSearchMethod (tree); //设置搜索方法
  norm_est.setKSearch (30); //设置最近邻的数量
  norm_est.setInputCloud (src); //设置输入云
  norm_est.compute (*points_with_normals_src); //计算法向量，并存储在points_with_normals_src
  pcl::copyPointCloud (*src, *points_with_normals_src); //复制点云（坐标）到points_with_normals_src（包含坐标和法向量）
  norm_est.setInputCloud (tgt); //这3行计算目标点云的法向量，同上
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  //创建一个 自定义点表达方式的 实例
  MyPointRepresentation point_representation;
  //加权曲率维度，以和坐标xyz保持平衡
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha); //设置缩放值（向量化点时使用）

  //创建非线性ICP对象 并设置参数
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg; //创建非线性ICP对象（ICP变体，使用Levenberg-Marquardt最优化）
  reg.setTransformationEpsilon (1e-6); //设置容许的最大误差（迭代最优化）
  //reg.setTransformationEpsilon (0.01); //设置容许的最大误差（迭代最优化）
  //***** 注意：根据自己数据库的大小调节该参数
  reg.setMaxCorrespondenceDistance (2);  //设置对应点之间的最大距离（2m）,在配准过程中，忽略大于该阈值的点  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation)); //设置点表达
  //设置源点云和目标点云
  reg.setInputSource (points_with_normals_src); //版本不符合，使用下面的语句
  //reg.setInputCloud (points_with_normals_src); //设置输入点云（待变换的点云）
  reg.setInputTarget (points_with_normals_tgt); //设置目标点云
  reg.setMaximumIterations (2); //设置内部优化的迭代次数

  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src; //用于存储结果（坐标+法向量）

  for (int i = 0; i < 100; ++i) //迭代
  {
    //pcl::ScopeTime scope_time("ICP Iteration"); 
    PCL_INFO ("Iteration Nr. %d.\n", i); //命令行显示迭代的次数
    //保存点云，用于可视化
    points_with_normals_src = reg_result; //
    //估计
    reg.setInputSource (points_with_normals_src);
    //reg.setInputCloud (points_with_normals_src); //重新设置输入点云（待变换的点云），因为经过上一次迭代，已经发生变换了
    reg.align (*reg_result); //对齐（配准）两个点云

    Ti = reg.getFinalTransformation () * Ti; //累积（每次迭代的）变换矩阵
    //如果这次变换和上次变换的误差比阈值小，通过减小最大的对应点距离的方法来进一步细化
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
       break;
     prev = reg.getLastIncrementalTransformation (); //上一次变换的误差
    //std::cout<<"getLastIncrementalTransformation"<<reg.getLastIncrementalTransformation ()<<endl;
    //std::cout<<"getLastIncrementalTransformation.sum: "<<reg.getLastIncrementalTransformation ().sum()<<endl;

    //显示当前配准状态，在窗口的右视区，简单的显示源点云和目标点云
    if(true == DEBUG_VISUALIZER)
    {
      showCloudsRight(points_with_normals_tgt, points_with_normals_src);    
    }
  }

  targetToSource = Ti.inverse(); //计算从目标点云到源点云的变换矩阵
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource); //将目标点云 变换回到 源点云帧

  //add the source to the transformed target
  *output += *cloud_src; // 拼接点云图（的点）点数数目是两个点云的点数和
  final_transform = targetToSource; //最终的变换。目标点云到源点云的变换矩阵

  if(true == DEBUG_VISUALIZER)
  {
    p->removePointCloud ("source"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
    p->removePointCloud ("target");
    p->removePointCloud("prePairAlign cloud");
    PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0); //设置点云显示颜色，下同
    PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
    p->addPointCloud (output, cloud_tgt_h, "target", vp_2); //添加点云数据，下同
    p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

    PCL_INFO ("Press q to continue the registration.\n");
    p->spin ();

    // p->removePointCloud ("source"); 
    // p->removePointCloud ("target");
    p->removePointCloud ("prePairAlign source"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
    p->removePointCloud ("prePairAlign target"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
    p->removeAllShapes();
  }
}

//订阅点云及可视化
void cloudCB(const sensor_msgs::PointCloud2& input)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud; // With color
 
    pcl::fromROSMsg(input, cloud); // sensor_msgs::PointCloud2 ----> pcl::PointCloud<T>
 
    p->removePointCloud ("source"); 
    p->removePointCloud ("target");
    p->addPointCloud<pcl::PointXYZRGB>(cloud.makeShared(), "cloud test");

    p->spin ();

    // //Convert the cloud to ROS message
    // pcl::toROSMsg(*source, output);
    // output.header.frame_id = "test";

    // ros::Rate loop_rate(1);
    // while (ros::ok())
    // {
    //     pcl_pub.publish(output);
    //     ros::spinOnce();
    //     loop_rate.sleep();
    // }


    //订阅相机发布点云数据
    // ros::Subscriber pcl_sub = nh.subscribe("/camera/depth_registered/points", 1, cloudCB);
    // ros::Rate rate(20.0);


    // pcl::toROSMsg(*CloudTransformedTarget, PointCloud2TransformedTarget);
    // PointCloud2TransformedTarget.header.frame_id = "CloudTransformedTarget";
    // ros::Rate loop_rate(1);

    // while (ros::ok())
    //   {
    //       pcl_pub.publish(PointCloud2TransformedTarget);
    //       ros::spinOnce();
    //       loop_rate.sleep();
    //   }
}

//std::string lable = "faceshoulders";
//std::string lable = "box_toothpaste_long";
std::string lable = "bottle_milktea";
bool bMaskRCNNMsg = true;

ros::Timer timer1;

void MaskRCNNCB(const sensor_msgs::ImageConstPtr& msg)
 {
   try
  {
    mask_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  if(false == bMaskRCNNMsg)
  {
      bMaskRCNNMsg = true;
      //depth_sub
  }
}

/***********************全局变量****************************/
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudMask(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudModel (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPreProcess (new pcl::PointCloud<pcl::PointXYZ>);
Eigen::Matrix4f  PairTransformation, GlobalTransformation = Eigen::Matrix4f::Identity(); 
PointCloud::Ptr CloudTransformedTarget (new PointCloud); //创建临时点云指针
int depth_cols = 640;
int depth_rows = 480;
int camera_factor = 1;
double camera_cx = 315.153;
double camera_cy = 253.73;
double camera_fx = 573.293;
double camera_fy = 572.41;

sensor_msgs::PointCloud2 PointCloud2TransformedTarget;

Eigen::Vector3d euler_Angle;
Eigen::Matrix3d Rotation_matrix = Eigen::Matrix3d::Identity();

void Alignment()
{
  //利用深度图和mask分割场景中物体点云 
  pcl::ScopeTime scope_time("**TotalAlignment");//计算算法运行时间
  for(int ImgWidth = 100; ImgWidth < depth_rows-100; ImgWidth++)
  {
    for(int ImgHeight = 200; ImgHeight < depth_cols-200; ImgHeight++ )
    {
      //获取深度图中对应点的深度值
      float d = cv_ptr->image.at<float>(ImgWidth,ImgHeight);
      //判断mask中是否是物体的点
      if(mask_ptr != 0)
      {
        unsigned char t = mask_ptr->image.at<unsigned char>(ImgWidth,ImgHeight);
        if(t == 0)
        continue;
      }
      else
      {
        if(d > 0.6)//单独测试，通过距离分割物体
        continue;
      }
      //计算这个点的空间坐标
      pcl::PointXYZ PointWorld;
      PointWorld.z = double(d)/camera_factor;
      PointWorld.x = (ImgHeight - camera_cx)*PointWorld.z/camera_fx;
      PointWorld.y = (ImgWidth - camera_cy)*PointWorld.z/camera_fy;
      CloudMask->points.push_back(PointWorld);
    }
  }
  //设置点云属性，采用无序排列方式存储点云
  CloudMask->height = 1;
  CloudMask->width = CloudMask->points.size();
  ROS_INFO("point cloud size = %d", CloudMask->width);
  CloudMask->is_dense = false;
  if(true == DEBUG_VISUALIZER)
  {
    p->removePointCloud ("target"); 
    p->removePointCloud ("source");
    p->addPointCloud<pcl::PointXYZ>(CloudMask, "cloud mask");
    p->spin();
  }
 
  //根据label提取物体模型
  //std::string lable = "faceshoulders";
  std::string ModelPath = "/home/model/catkin_ws2/src/pose_estimation/model_pcd/";
  ModelPath = ModelPath + lable + "_model.pcd";
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (ModelPath, *CloudModel) == -1) 
  {
    PCL_ERROR ("Couldn't read file bottle_milktea_model.pcd \n");
    return;
  }    
  ROS_INFO("points loaded from Model = %d",  CloudModel->width * CloudModel->height);
 
  if(true == DEBUG_VISUALIZER)
  {
    showCloudsLeft(CloudMask, CloudModel); //在左视区，简单的显示源点云和目标点云
  }     
  //配准物体模型和场景中物体点云
  {
    pcl::ScopeTime scope_time("*PrePairAlign");//计算算法运行时间
    prePairAlign(CloudMask,CloudModel,CloudPreProcess,true);
  }
  if(true == DEBUG_VISUALIZER)
  {
    PCL_INFO ("Press q to continue.\n");
    p->spin();
  }
  {
    pcl::ScopeTime scope_time("*PairAlign");//计算算法运行时间
    pairAlign (CloudMask, CloudPreProcess, CloudTransformedTarget, PairTransformation, true);
    bMaskRCNNMsg = false;
  }
  std::cout << "The Estimated Rotation and translation matrices are : \n" << std::endl;
  printf("\n");
  printf("    | %6.3f %6.3f %6.3f | \n", PairTransformation(0, 0), PairTransformation(0, 1), PairTransformation(0, 2));
  printf("R = | %6.3f %6.3f %6.3f | \n", PairTransformation(1, 0), PairTransformation(1, 1), PairTransformation(1, 2));
  printf("    | %6.3f %6.3f %6.3f | \n", PairTransformation(2, 0), PairTransformation(2, 1), PairTransformation(2, 2));
  printf("\n");
  printf("t = < %0.3f, %0.3f, %0.3f >\n", PairTransformation(0, 3), PairTransformation(1, 3), PairTransformation(2, 3));

  GlobalTransformation = PairTransformation*transformation2;
  std::cout << "The Global Rotation and translation matrices are : \n" << std::endl;
  printf("\n");
  printf("    | %6.3f %6.3f %6.3f | \n", GlobalTransformation(0, 0), GlobalTransformation(0, 1), GlobalTransformation(0, 2));
  printf("R = | %6.3f %6.3f %6.3f | \n", GlobalTransformation(1, 0), GlobalTransformation(1, 1), GlobalTransformation(1, 2));
  printf("    | %6.3f %6.3f %6.3f | \n", GlobalTransformation(2, 0), GlobalTransformation(2, 1), GlobalTransformation(2, 2));
  printf("\n");
  printf("t = < %0.3f, %0.3f, %0.3f >\n", GlobalTransformation(0, 3), GlobalTransformation(1, 3), GlobalTransformation(2, 3));
  if(true == DEBUG_VISUALIZER)
    {
      PCL_INFO ("Press q to contin.\n");
      p->spin();
    }
}

// void timer1(const ros::TimerEvent& event)
// {
//   std::cout<<"timer1"<<endl;
// }

  double thetax = 0;
  double thetay = 0;
  double thetaz = 0;

//depth图显示的回调函数    
void depthCb(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  // //可视化深度图
  // if(true == DEBUG_VISUALIZER)
  // {
  //   //Draw an example circle on the video stream
  //   if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
  //     cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
  //   //Update GUI Window
  //   cv::imshow("depth image", cv_ptr->image);
  //   cv::waitKey(3);
  // }

  //timer1 = nh.createTimer(ros::Duration(0.1), timer1, true);
  if(true == bMaskRCNNMsg)
  {
    depth_sub.shutdown();
    cv::waitKey(100);
    Alignment();
    bMaskRCNNMsg = false;  
  }

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
     {
        Rotation_matrix(i, j) = GlobalTransformation(i, j);
     }
  euler_Angle = Rotation_matrix.eulerAngles(2, 1, 0);//顺序Z, Y, X
  thetax = euler_Angle[2];
  thetay = euler_Angle[1];
  thetaz = euler_Angle[0];
  std::cout<<thetax<<endl;
  std::cout<<thetay<<endl;
  std::cout<<thetaz<<endl;

}

void Pose_Visualer(const ros::TimerEvent& event)
{
  tf::Transform transform;
  static tf::TransformBroadcaster br;
  transform.setOrigin (tf::Vector3(GlobalTransformation(0, 3), GlobalTransformation(1, 3), GlobalTransformation(2, 3)));
  transform.setRotation (tf::createQuaternionFromRPY (thetax, thetay, thetaz));// X Y Z
  br.sendTransform (tf::StampedTransform(transform, ros::Time::now (),"/camera_rgb_optical_frame", "/object"));
  //[Right to Left Camera Translate]
  //-25.0752 0.155353 0.384843 
}

  

//****************  主函数  ************************
int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pcl_registration");
  ros::NodeHandle nh;

  // Rviz物体姿态可视化timer
  ros::Timer timer = nh.createTimer(ros::Duration(0.1), Pose_Visualer);
  sensor_msgs::PointCloud2 output;
  pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);//发布到主题（topic）

  if(4 == argc)
  {
    if(strcmp(argv[1], "-v") == 0)
    {
      DEBUG_VISUALIZER = true; 
      cout << "Visualizer = " << "true" << endl;
    }
    else
      cout << "Visualizer = " << "false" << endl;
    
    if(strcmp(argv[2], "-m") == 0)
      lable = argv[3];
  }
  else if(2 == argc)
  {
    if(strcmp(argv[1], "-v") == 0)
    {
      DEBUG_VISUALIZER = true; 
      cout << "Visualizer = " << "true" << endl;
    }
    else
      cout << "Visualizer = " << "false" << endl;
  }
  else
  {
    DEBUG_VISUALIZER = false; 
    cout << "Visualizer = " << "false" << endl;
  }
  
  if(true == DEBUG_VISUALIZER)
  {
    //创建一个 PCLVisualizer 对象
    p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Registration"); //p是全局变量
    p->createViewPort (0.0, 0, 0.5, 1.0, vp_1); //创建左视区
    p->createViewPort (0.5, 0, 1.0, 1.0, vp_2); //创建右视区
    //cv::namedWindow("depth image");
  }
  //创建点云指针和变换矩阵
  PointCloud::Ptr result (new PointCloud), source, pretarget(new PointCloud), target; //创建3个点云指针，分别用于结果，源点云和目标点云
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform; 

  //ROS下与其他节点共同集成测试
  if(true == DEBUG_WITH_OTHERS)
  {
    //订阅深度图
    depth_sub = nh.subscribe("/camera/depth_registered/sw_registered/image_rect", 1 , depthCb); 
    //订阅mask-RCNN发布消息
    MaskRCNN_sub = nh.subscribe("maskRCNN",1,MaskRCNNCB); 
    //cv::destroyWindow("depth image"); 
  }
  else 
  {
    //模块独立测试，从硬盘读取测试点云和模型，
    //格式为data[0]=测试点云1
    //data[1]=物体模型1
    //data[2]=测试点云2
    //data[3]=物体点云2
    // ……
    std::vector<PCD, Eigen::aligned_allocator<PCD> > data; //模型
    loadData (argc, argv, data); //读取pcd文件数据，定义见上面
    //检查用户数据
    if (data.empty ())
    {
      PCL_ERROR ("Syntax is: %s <source.pcd> <target.pcd> [*]", argv[0]); //语法
      PCL_ERROR ("[*] - multiple files can be added. The registration results of (i, i+1) will be registered against (i+2), etc"); //可以使用多个文件
      return (-1);
    }
    PCL_INFO ("Loaded %d datasets.", (int)data.size ()); //显示读取了多少个点云文件
      //遍历所有的点云文件
    for (size_t i = 1; i < data.size (); ++i, ++i)
    {
      source = data[i-1].cloud; //源点云
      target = data[i].cloud; //目标点云
      showCloudsLeft(source, target); //在左视区，简单的显示源点云和目标点云
      PointCloud::Ptr temp (new PointCloud); //创建临时点云指针
          //显示正在配准的点云文件名和各自的点数
      PCL_INFO ("Aligning %s (%d points) with %s (%d points).\n", data[i-1].f_name.c_str (), source->points.size (), data[i].f_name.c_str (), target->points.size ());

      prePairAlign(source, target, pretarget, true);
      pairAlign (source, target, temp, pairTransform, true);
      //将当前的一对点云数据，变换到全局变换中。
      pcl::transformPointCloud (*temp, *result, GlobalTransform);
      //更新全局变换
      GlobalTransform = GlobalTransform * pairTransform;
    }
  }

  // ros::MultiThreadedSpinner spinner(4); // Use 4 threads
  // spinner.spin(); 
  // ros::Rate r(10); // 10 hz
  // while (true)
  // {
  //   ros::spinOnce();
  //   r.sleep();
  // }
  ros::spin ();
}

