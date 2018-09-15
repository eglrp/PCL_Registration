
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
//ros std消息
#include "std_msgs/String.h"
#include "std_msgs/Int8.h"

#include <pcl/segmentation/extract_clusters.h>

//pcl类型名简化
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

//**************全局变量**************//
//可视化对象
pcl::visualization::PCLVisualizer *p;
//左视区和右视区，可视化窗口分成左右两部分
int vp_1, vp_2;
//ROS图像转OpenCV变量
cv_bridge::CvImagePtr depth_ptr, mask_ptr;
//调试标志
bool DEBUG_VISUALIZER = false;
//prePairAlign transformation matrix
pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 MomentOfInertia_Transformation;
//PairAlign transformation matrix
Eigen::Matrix4f  PairAlign_Transformation, GlobalTransformation = Eigen::Matrix4f::Identity(); 

//在窗口的左视区，简单的显示源点云和目标点云
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
  p->removePointCloud ("vp1_source"); //
  pcl::visualization::PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0); //目标点云绿色
  pcl::visualization::PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0); //源点云红色
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
  pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature"); //目标点云彩色句柄
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");
  pcl::visualization::PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature"); //源点云彩色句柄
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");
  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2); //加载点云
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);
  p->spinOnce();
}

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
  else //不进行下采样
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

	TESVD.estimateRigidTransformation(*cloud_in, *cloud_out, MomentOfInertia_Transformation);
	//输出变换矩阵信息  
	std::cout << "The Pre-estimated Rotation and translation matrices are : \n" << std::endl;
	printf("\n");
	printf("    | %6.3f %6.3f %6.3f | \n", MomentOfInertia_Transformation(0, 0), MomentOfInertia_Transformation(0, 1), MomentOfInertia_Transformation(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", MomentOfInertia_Transformation(1, 0), MomentOfInertia_Transformation(1, 1), MomentOfInertia_Transformation(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", MomentOfInertia_Transformation(2, 0), MomentOfInertia_Transformation(2, 1), MomentOfInertia_Transformation(2, 2));
	printf("\n");
	printf("t = < %0.3f, %0.3f, %0.3f >\n", MomentOfInertia_Transformation(0, 3), MomentOfInertia_Transformation(1, 3), MomentOfInertia_Transformation(2, 3));

	//Executing the transformation
	//pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
	//You can either apply transform_1 or transform_2; they are the same
  //pcl::transformPointCloud(*cloud_src, *transformed_cloud, MomentOfInertia_Transformation);
  pcl::transformPointCloud(*cloud_tgt, *transformed_cloud, MomentOfInertia_Transformation);
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
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0); //设置点云显示颜色，下同
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
    p->addPointCloud (output, cloud_tgt_h, "target", vp_2); //添加点云数据，下同
    p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

    PCL_INFO ("Press q to continue the registration.\n");
    p->spin ();

    p->removePointCloud ("prePairAlign source"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
    p->removePointCloud ("prePairAlign target"); //根据给定的ID，从屏幕中去除一个点云。参数是ID
    p->removeAllShapes();
  }
}

void EuclideanCluster(const PointCloud::Ptr cloud_Segmentation, const PointCloud::Ptr cloud_EuclideanCluster)
{
//********************************欧式聚类******************************//
	//创建用于提取搜索方法的kdtree树对象
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud_Segmentation);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;   //欧式聚类对象
	ec.setClusterTolerance(0.02);                     // 设置近邻搜索的搜索半径为2cm
	ec.setMinClusterSize(100);                 //设置一个聚类需要的最少的点数目为10
	ec.setMaxClusterSize(1000000);               //设置一个聚类需要的最大点数目为250000
	ec.setSearchMethod(tree);                    //设置点云的搜索机制
	ec.setInputCloud(cloud_Segmentation);
	ec.extract(cluster_indices);           //从点云中提取聚类，并将点云索引保存在cluster_indices中
										   //迭代访问点云索引cluster_indices,直到分割处所有聚类
	std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
	//容器中的点云的索引第一个为物体点云数据
	//pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		//设置保存点云的属性问题
	cloud_EuclideanCluster->points.push_back(cloud_Segmentation->points[*pit]);
	cloud_EuclideanCluster->width = cloud_EuclideanCluster->points.size();
	cloud_EuclideanCluster->height = 1;
	cloud_EuclideanCluster->is_dense = true;
	std::cout << "PointCloud representing the Cluster: " << cloud_EuclideanCluster->points.size() << " data points." << std::endl;
}

/***********************全局变量****************************/
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudMask(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudModel (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPreProcess (new pcl::PointCloud<pcl::PointXYZ>);

pcl::PointCloud<pcl::PointXYZ>::Ptr CloudEuclideanCluster(new pcl::PointCloud<pcl::PointXYZ>);

PointCloud::Ptr CloudTransformedTarget (new PointCloud); //创建临时点云指针
int depth_cols = 640;
int depth_rows = 480;
int camera_factor = 1;
double camera_cx = 315.153;
double camera_cy = 253.73;
double camera_fx = 573.293;
double camera_fy = 572.41;

double thetax = 0;
double thetay = 0;
double thetaz = 0;

Eigen::Vector3d euler_Angle;
Eigen::Matrix3d Rotation_matrix = Eigen::Matrix3d::Identity();

std::string label = "bottle_milktea";
bool bUpdatingImage = false;
bool bSaveImage = false;

sensor_msgs::PointCloud2 PointCloud2TransformedTarget;

//整合的配准函数
void Alignment()
{
  //利用深度图和mask分割场景中物体点云 
  pcl::ScopeTime scope_time("**TotalAlignment");//计算算法运行时间
  for(int ImgWidth = 100; ImgWidth < depth_rows-100; ImgWidth++)
  {
    for(int ImgHeight = 200; ImgHeight < depth_cols-200; ImgHeight++ )
    {
      //获取深度图中对应点的深度值
      float d = depth_ptr->image.at<float>(ImgWidth,ImgHeight);
      //判断mask中是否是物体的点
      if(mask_ptr != 0)
      {
        unsigned char t = mask_ptr->image.at<unsigned char>(ImgWidth,ImgHeight);
        if(t == 0)
        continue;
      }
      else
      {
        ROS_INFO("mask image pointer mask_ptr = null");
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
  ROS_INFO("mask cloud size = %d", CloudMask->width);
  CloudMask->is_dense = false;
  if(true == DEBUG_VISUALIZER)
  {
    p->removePointCloud ("target"); 
    p->removePointCloud ("source");
    p->addPointCloud<pcl::PointXYZ>(CloudMask, "cloud mask");
    p->spin();
  }
 
  //欧式聚类去处理离群点，保留最大点集，避免RGB—D对齐误差或者MaskRCNN识别误差导致分层现象
  //EuclideanCluster(CloudMask, CloudEuclideanCluster);

  //根据label提取物体模型
  //std::string ModelPath = "/home/siasun/Desktop/RobGrab/src/pose_estimation/model_pcd/";
  std::string ModelPath = "/home/model/catkin_ws2/src/pose_estimation/model_pcd/";
  ModelPath = ModelPath + label + "_model.pcd";
  std::cout << "Object Label : " << label << endl;
  if (pcl::io::loadPCDFile<pcl::PointXYZ> (ModelPath, *CloudModel) == -1) 
  {
    //PCL_ERROR ("Couldn't read file bottle_milktea_model.pcd \n");
    std::cout << "Couldn't read file " << ModelPath <<endl;
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
    pairAlign (CloudMask, CloudPreProcess, CloudTransformedTarget, PairAlign_Transformation, true);
    bSaveImage = false;
  }
  std::cout << "The Estimated Rotation and translation matrices are : \n" << std::endl;
  printf("\n");
  printf("    | %6.3f %6.3f %6.3f | \n", PairAlign_Transformation(0, 0), PairAlign_Transformation(0, 1), PairAlign_Transformation(0, 2));
  printf("R = | %6.3f %6.3f %6.3f | \n", PairAlign_Transformation(1, 0), PairAlign_Transformation(1, 1), PairAlign_Transformation(1, 2));
  printf("    | %6.3f %6.3f %6.3f | \n", PairAlign_Transformation(2, 0), PairAlign_Transformation(2, 1), PairAlign_Transformation(2, 2));
  printf("\n");
  printf("t = < %0.3f, %0.3f, %0.3f >\n", PairAlign_Transformation(0, 3), PairAlign_Transformation(1, 3), PairAlign_Transformation(2, 3));

  GlobalTransformation = PairAlign_Transformation*MomentOfInertia_Transformation;
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

/*
 * Class Listener use Boost.Bind to pass arbitrary data into a subscription
 * callback.  For more information on Boost.Bind see the documentation on the boost homepage,
 * http://www.boost.org/
 */
class Listener
{
  public:
  ros::NodeHandle node_handle_;
  ros::V_Subscriber subs_;    //std::vector<Subscriber>  向量容器
  
  ros::Subscriber depth_sub_;
  ros::Subscriber MaskRCNN_sub_;
  ros::Subscriber RobotControl_sub_;
  ros::Subscriber Label_sub_;

  Listener(const ros::NodeHandle& node_handle)
  : node_handle_(node_handle) //冒号的含义是使用参数node_handle对类的成员node_handle_进行初始化
  {
  }

  void init()  //Listener类的init()方法
  {
    // std::vector<Subscriber>.push_back()在容器尾部加入一个Subscriber对象(节点句柄的subscribe方法返回的) ，
    // boost::bind方法将一个函数转化成另一个函数
    //subs_.push_back(node_handle_.subscribe<std_msgs::String>("chatter", 1000, boost::bind(&Listener::chatterCallback, this, _1, "User 1")));
    subs_.push_back(node_handle_.subscribe<sensor_msgs::Image>("/segment/segment_image", 1, boost::bind(&Listener::Mask_Callback, this, _1, node_handle_)));
    //订阅RC消息，采集图像
    RobotControl_sub_ = node_handle_.subscribe("/command/recog_command", 1, CaptureImage_Callback);
    //订阅深度图
    depth_sub_ = node_handle_.subscribe("/camera/depth_registered/sw_registered/image_rect", 1 , Depth_Callback);
    //订阅label
    Label_sub_ = node_handle_.subscribe("/segment/segment_class", 1, Label_Callback); 
  }
  // void chatterCallback(const std_msgs::String::ConstPtr& msg, std::string user_string)  //被boost::bind()转化之前的消息回调函数
  // {
  //   ROS_INFO("I heard: [%s] with user string [%s]", msg->data.c_str(), user_string.c_str());
  // }
  void static Depth_Callback(const sensor_msgs::ImageConstPtr& msg);
  void static Cloud_Callback(const sensor_msgs::PointCloud2& input);
  void static Label_Callback(const std_msgs::String::ConstPtr& msg);
  void static CaptureImage_Callback(const std_msgs::Int8::ConstPtr& msg);
  void Mask_Callback(const sensor_msgs::ImageConstPtr& msg, ros::NodeHandle& node_handle);
};

//depth图显示的回调函数    
void Listener::Depth_Callback(const sensor_msgs::ImageConstPtr& msg)
{
  if(true == bSaveImage)
  {
     bSaveImage = false; 
     bUpdatingImage = true;
     ROS_INFO("Start saving depth image");
     try
    {
      depth_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    bUpdatingImage = false;
    ROS_INFO("Stop saving depth image");
  // //可视化深度图
  // if(true == DEBUG_VISUALIZER)
  // {
  //   //Draw an example circle on the video stream
  //   if (depth_ptr->image.rows > 60 && depth_ptr->image.cols > 60)
  //     cv::circle(depth_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
  //   //Update GUI Window
  //   cv::imshow("depth image", depth_ptr->image);
  //   cv::waitKey(3);
  // }
  }
}

void Listener::Mask_Callback(const sensor_msgs::ImageConstPtr& msg, ros::NodeHandle& node_handle)
{
  ROS_INFO("Mask Callback");
  try
  {
    mask_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  
  ros::Rate r(10); // 10 hz
  //判断深度图是否正在更新
  while(bUpdatingImage)//********************需要验证是否阻塞其他订阅会掉线程*************************//
  {
    ROS_INFO("Waiting for saving depth image");
    ros::spinOnce();
    r.sleep();
  }
  //关闭相机深度图订阅，1、锁定准备操作的深度图；2、减少配准过程订阅图像的资源利用率
  //depth_sub.shutdown();
  this->depth_sub_.shutdown();
  //深度图和mask都已准备就绪，进行匹配
  ROS_INFO("Start Alignment");
  Alignment();
  //回复订阅相机深度图
  this->depth_sub_ = node_handle.subscribe("/camera/depth_registered/sw_registered/image_rect", 1 , Depth_Callback);//******需要验证是否能够重新订阅成功*******//
}

//订阅点云及可视化
void Listener::Cloud_Callback(const sensor_msgs::PointCloud2& input)
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
    // ros::Subscriber pcl_sub = ros_nodehandle.subscribe("/camera/depth_registered/points", 1, Cloud_Callback);
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

void Listener::Label_Callback(const std_msgs::String::ConstPtr& msg)
{
  ROS_INFO("Label Callback");
  ROS_INFO("I heard: [%s]", msg->data.c_str());
  label =  msg->data.c_str();
}

void Listener::CaptureImage_Callback(const std_msgs::Int8::ConstPtr& msg)
{
  ROS_INFO("RobotControl Callback");
  if(1 == msg->data)
  {
    if(false == bSaveImage)
    {
        bSaveImage = true;
    }
  }
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
  ros::NodeHandle ros_nodehandle;
  // Rviz物体姿态可视化timer
  ros::Timer timer = ros_nodehandle.createTimer(ros::Duration(0.1), Pose_Visualer);
  //ros发布对象
  ros::Publisher pcl_pub; 
  sensor_msgs::PointCloud2 output;
  pcl_pub = ros_nodehandle.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);//发布到主题（pcl_output）
  
  //输入参数判断，“-v”可视化算法过程
  if(2 == argc)
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
  //可视化调试过程
  if(true == DEBUG_VISUALIZER)
  {
    p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Registration"); //创建一个 PCLVisualizer 对象，p是全局变量
    p->createViewPort (0.0, 0, 0.5, 1.0, vp_1); //创建左视区
    p->createViewPort (0.5, 0, 1.0, 1.0, vp_2); //创建右视区
  }

  Listener listener(ros_nodehandle);        //创建Listener类
  listener.init();                          //调用Listener类的init（）方法，创建3个话题订阅者，并压入容器
 
  ros::spin ();
}
