
/**************************************************
Copyright(c) 2018-2018 fellen All rights reserved. 
Author: fellen
Date:2018-08-20 
Description:1��Compute starting translation and rotation based on MomentOfInertiaEstimation descriptor
            2��LM-ICP Alignment                                                                                                                                          
**************************************************/

//ROSͷ�ļ�
#include <ros/ros.h>
//ROS���ݸ�ʽ��PCL���ݸ�ʽת��
#include <pcl_conversions/pcl_conversions.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
//boostָ�����
#include <boost/make_shared.hpp> 
//��/����
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
//pcd�ļ�����/���
#include <pcl/io/pcd_io.h>
//�˲�
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
//����
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
//��׼
#include <pcl/registration/icp.h> 
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
//���ӻ�
#include <pcl/visualization/pcl_visualizer.h>
//opencv2
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//image transport
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
//�㷨ʱ�����
#include <pcl/common/time.h>
//�ַ���
#include <string>
//tf����ϵ�任
#include <tf/transform_broadcaster.h>

//�����ռ�
using pcl::visualization::PointCloudColorHandlerGenericField;
using pcl::visualization::PointCloudColorHandlerCustom;

//�������͵ı���
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

//**************ȫ�ֱ���**************//
//ros��������
ros::Publisher pub;
ros::Subscriber depth_sub;
ros::Subscriber MaskRCNN_sub;

ros::Publisher pcl_pub; 

//���ӻ�����
pcl::visualization::PCLVisualizer *p;
//�������������������ӻ����ڷֳ�����������
int vp_1, vp_2;
//ROSͼ��תOpenCV����
cv_bridge::CvImagePtr cv_ptr, mask_ptr;
//���Ա�־
bool DEBUG_WITH_OTHERS = true;
bool DEBUG_VISUALIZER = false;

//����ṹ�壬���ڴ������
struct PCD
{
  PointCloud::Ptr cloud; //����ָ��
  std::string f_name; //�ļ���
  PCD() : cloud (new PointCloud) {}; //���캯����ʼ��
};

// �����µĵ��﷽ʽ< x, y, z, curvature > ����+����
class MyPointRepresentation : public pcl::PointRepresentation <PointNormalT> //�̳й�ϵ
{
  using pcl::PointRepresentation<PointNormalT>::nr_dimensions_;
  public:
  MyPointRepresentation ()
  {
    //ָ��ά��
    nr_dimensions_ = 4;
  }
  //���غ���copyToFloatArray���Զ����Լ�����������
  virtual void copyToFloatArray (const PointNormalT &p, float * out) const
  {
    //< x, y, z, curvature > ����xyz������
    out[0] = p.x;
    out[1] = p.y;
    out[2] = p.z;
    out[3] = p.curvature;
  }
};

//�ڴ��ڵ����������򵥵���ʾԴ���ƺ�Ŀ�����
void showCloudsLeft(const PointCloud::Ptr cloud_target, const PointCloud::Ptr cloud_source)
{
  p->removePointCloud ("vp1_target"); //���ݸ�����ID������Ļ��ȥ��һ�����ơ�������ID
  p->removePointCloud ("vp1_source"); //
  PointCloudColorHandlerCustom<PointT> tgt_h (cloud_target, 0, 255, 0); //Ŀ�������ɫ
  PointCloudColorHandlerCustom<PointT> src_h (cloud_source, 255, 0, 0); //Դ���ƺ�ɫ
  p->addPointCloud (cloud_target, tgt_h, "vp1_target", vp_1); //���ص���
  p->addPointCloud (cloud_source, src_h, "vp1_source", vp_1);
  PCL_INFO ("Press q to begin the registration.\n"); //������������ʾ��ʾ��Ϣ
  p-> spin();
}

//�ڴ��ڵ����������򵥵���ʾԴ���ƺ�Ŀ�����
void showCloudsRight(const PointCloudWithNormals::Ptr cloud_target, const PointCloudWithNormals::Ptr cloud_source)
{
  p->removePointCloud ("source"); //���ݸ�����ID������Ļ��ȥ��һ�����ơ�������ID
  p->removePointCloud ("target");
  PointCloudColorHandlerGenericField<PointNormalT> tgt_color_handler (cloud_target, "curvature"); //Ŀ����Ʋ�ɫ���
  if (!tgt_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");
  PointCloudColorHandlerGenericField<PointNormalT> src_color_handler (cloud_source, "curvature"); //Դ���Ʋ�ɫ���
  if (!src_color_handler.isCapable ())
      PCL_WARN ("Cannot create curvature color handler!");
  p->addPointCloud (cloud_target, tgt_color_handler, "target", vp_2); //���ص���
  p->addPointCloud (cloud_source, src_color_handler, "source", vp_2);
  p->spinOnce();
}

// ��ȡһϵ�е�PCD�ļ���ϣ����׼�ĵ����ļ���
// ����argc ����������������main()��
// ����argv �������б�����main()��
// ����models �������ݼ��Ľ������
void loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  std::string extension (".pcd"); //��������ʼ��string���ͱ���extension����ʾ�ļ���׺��
  // ͨ�������ļ�������ȡpcd�ļ�
  for (int i = 1; i < argc; i++) //�������е��ļ������Թ���������
  {
    std::string fname = std::string (argv[i]);
    if (fname.size () <= extension.size ()) //�ļ����ĳ����Ƿ����Ҫ��
      continue;

    std::transform (fname.begin (), fname.end (), fname.begin (), (int(*)(int))tolower); //��ĳ����(Сд��ĸ��)Ӧ����ָ����Χ��ÿ��Ԫ��
    //����ļ��Ƿ���pcd�ļ�
    if (fname.compare (fname.size () - extension.size (), extension.size (), extension) == 0)
    {
      // ��ȡ���ƣ������浽models
      PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloud); //��ȡ��������
      //ȥ�������е�NaN�㣨xyz����NaN��
      std::vector<int> indices; //����ȥ���ĵ������
      pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices); //ȥ�������е�NaN��
      models.push_back (m);
    }
  }
}


pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ>::Matrix4 transformation2;
void prePairAlign(const PointCloud::Ptr cloud_src,const PointCloud::Ptr cloud_tgt, PointCloud::Ptr transformed_cloud,  bool downsample)
{
  PointCloud::Ptr src (new PointCloud); //��������ָ��
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid; //VoxelGrid ��һ�������ĵ��ƣ��ۼ���һ���ֲ���3D������,���²������˲���������
  if (downsample) //�²���
  {
    grid.setLeafSize (0.007, 0.007, 0.007); //������Ԫ�����Ҷ�Ӵ�С
        //�²��� Դ����
    grid.setInputCloud (cloud_src); //�����������
    grid.filter (*src); //�²������˲������洢��src��
        //�²��� Ŀ�����
    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);

    PCL_INFO ("Partial Pointcloud size after sampling is. %d.\n", src->size());
    PCL_INFO ("Model Pointcloud size after sampling is. %d.\n", tgt->size());
  }
  else //���²���
  {
    src = cloud_src; //ֱ�Ӹ���
    tgt = cloud_tgt;
  }
  //******************************OBB��Χ�м���*************************************//
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

    //***********************************���ӻ����ġ���Χ�к�����ϵ******************************************//
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


	//************************************������ת����***********************************//
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

	//�����ĸ���
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

	//Ŀ���ĸ���
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

	//����SVD�������任����  
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> TESVD;

	TESVD.estimateRigidTransformation(*cloud_in, *cloud_out, transformation2);
	//����任������Ϣ  
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

//�򵥵���׼һ�Ե������ݣ������ؽ��
//����cloud_src  Դ����
//����cloud_tgt  Ŀ�����
//����output     �������
//����final_transform �ɶԱ任����
//����downsample �Ƿ��²���
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
{
  //Ϊ��һ���Ժ��ٶȣ��²���
  PointCloud::Ptr src (new PointCloud); //��������ָ��
  PointCloud::Ptr tgt (new PointCloud);
  pcl::VoxelGrid<PointT> grid; //VoxelGrid ��һ�������ĵ��ƣ��ۼ���һ���ֲ���3D������,���²������˲���������
  if (downsample) //�²���
  {
    grid.setLeafSize (0.007, 0.007, 0.007); //������Ԫ�����Ҷ�Ӵ�С
        //�²��� Դ����
    grid.setInputCloud (cloud_src); //�����������
    grid.filter (*src); //�²������˲������洢��src��
        //�²��� Ŀ�����
    grid.setInputCloud (cloud_tgt);
    grid.filter (*tgt);
  }
  else //���²���
  {
    src = cloud_src; //ֱ�Ӹ���
    tgt = cloud_tgt;
  }

  //��������ķ�����������
  PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals); //����Դ����ָ�루ע�������Ͱ�������ͷ�������
  PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals); //����Ŀ�����ָ�루ע�������Ͱ�������ͷ�������
  pcl::NormalEstimation<PointT, PointNormalT> norm_est; //�ö������ڼ��㷨����
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ()); //����kd�������ڼ��㷨��������������
  norm_est.setSearchMethod (tree); //������������
  norm_est.setKSearch (30); //��������ڵ�����
  norm_est.setInputCloud (src); //����������
  norm_est.compute (*points_with_normals_src); //���㷨���������洢��points_with_normals_src
  pcl::copyPointCloud (*src, *points_with_normals_src); //���Ƶ��ƣ����꣩��points_with_normals_src����������ͷ�������
  norm_est.setInputCloud (tgt); //��3�м���Ŀ����Ƶķ�������ͬ��
  norm_est.compute (*points_with_normals_tgt);
  pcl::copyPointCloud (*tgt, *points_with_normals_tgt);

  //����һ�� �Զ�����﷽ʽ�� ʵ��
  MyPointRepresentation point_representation;
  //��Ȩ����ά�ȣ��Ժ�����xyz����ƽ��
  float alpha[4] = {1.0, 1.0, 1.0, 1.0};
  point_representation.setRescaleValues (alpha); //��������ֵ����������ʱʹ�ã�

  //����������ICP���� �����ò���
  pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg; //����������ICP����ICP���壬ʹ��Levenberg-Marquardt���Ż���
  reg.setTransformationEpsilon (1e-6); //���������������������Ż���
  //reg.setTransformationEpsilon (0.01); //���������������������Ż���
  //***** ע�⣺�����Լ����ݿ�Ĵ�С���ڸò���
  reg.setMaxCorrespondenceDistance (2);  //���ö�Ӧ��֮��������루2m��,����׼�����У����Դ��ڸ���ֵ�ĵ�  reg.setPointRepresentation (boost::make_shared<const MyPointRepresentation> (point_representation)); //���õ���
  //����Դ���ƺ�Ŀ�����
  reg.setInputSource (points_with_normals_src); //�汾�����ϣ�ʹ����������
  //reg.setInputCloud (points_with_normals_src); //����������ƣ����任�ĵ��ƣ�
  reg.setInputTarget (points_with_normals_tgt); //����Ŀ�����
  reg.setMaximumIterations (2); //�����ڲ��Ż��ĵ�������

  // Run the same optimization in a loop and visualize the results
  Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity (), prev, targetToSource;
  PointCloudWithNormals::Ptr reg_result = points_with_normals_src; //���ڴ洢���������+��������

  for (int i = 0; i < 100; ++i) //����
  {
    //pcl::ScopeTime scope_time("ICP Iteration"); 
    PCL_INFO ("Iteration Nr. %d.\n", i); //��������ʾ�����Ĵ���
    //������ƣ����ڿ��ӻ�
    points_with_normals_src = reg_result; //
    //����
    reg.setInputSource (points_with_normals_src);
    //reg.setInputCloud (points_with_normals_src); //��������������ƣ����任�ĵ��ƣ�����Ϊ������һ�ε������Ѿ������任��
    reg.align (*reg_result); //���루��׼����������

    Ti = reg.getFinalTransformation () * Ti; //�ۻ���ÿ�ε����ģ��任����
    //�����α任���ϴα任��������ֵС��ͨ����С���Ķ�Ӧ�����ķ�������һ��ϸ��
    if (fabs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
       break;
     prev = reg.getLastIncrementalTransformation (); //��һ�α任�����
    //std::cout<<"getLastIncrementalTransformation"<<reg.getLastIncrementalTransformation ()<<endl;
    //std::cout<<"getLastIncrementalTransformation.sum: "<<reg.getLastIncrementalTransformation ().sum()<<endl;

    //��ʾ��ǰ��׼״̬���ڴ��ڵ����������򵥵���ʾԴ���ƺ�Ŀ�����
    if(true == DEBUG_VISUALIZER)
    {
      showCloudsRight(points_with_normals_tgt, points_with_normals_src);    
    }
  }

  targetToSource = Ti.inverse(); //�����Ŀ����Ƶ�Դ���Ƶı任����
  pcl::transformPointCloud (*cloud_tgt, *output, targetToSource); //��Ŀ����� �任�ص� Դ����֡

  //add the source to the transformed target
  *output += *cloud_src; // ƴ�ӵ���ͼ���ĵ㣩������Ŀ���������Ƶĵ�����
  final_transform = targetToSource; //���յı任��Ŀ����Ƶ�Դ���Ƶı任����

  if(true == DEBUG_VISUALIZER)
  {
    p->removePointCloud ("source"); //���ݸ�����ID������Ļ��ȥ��һ�����ơ�������ID
    p->removePointCloud ("target");
    p->removePointCloud("prePairAlign cloud");
    PointCloudColorHandlerCustom<PointT> cloud_tgt_h (output, 0, 255, 0); //���õ�����ʾ��ɫ����ͬ
    PointCloudColorHandlerCustom<PointT> cloud_src_h (cloud_src, 255, 0, 0);
    p->addPointCloud (output, cloud_tgt_h, "target", vp_2); //��ӵ������ݣ���ͬ
    p->addPointCloud (cloud_src, cloud_src_h, "source", vp_2);

    PCL_INFO ("Press q to continue the registration.\n");
    p->spin ();

    // p->removePointCloud ("source"); 
    // p->removePointCloud ("target");
    p->removePointCloud ("prePairAlign source"); //���ݸ�����ID������Ļ��ȥ��һ�����ơ�������ID
    p->removePointCloud ("prePairAlign target"); //���ݸ�����ID������Ļ��ȥ��һ�����ơ�������ID
    p->removeAllShapes();
  }
}

//���ĵ��Ƽ����ӻ�
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


    //�������������������
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

/***********************ȫ�ֱ���****************************/
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudMask(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudModel (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr CloudPreProcess (new pcl::PointCloud<pcl::PointXYZ>);
Eigen::Matrix4f  PairTransformation, GlobalTransformation = Eigen::Matrix4f::Identity(); 
PointCloud::Ptr CloudTransformedTarget (new PointCloud); //������ʱ����ָ��
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
  //�������ͼ��mask�ָ����������� 
  pcl::ScopeTime scope_time("**TotalAlignment");//�����㷨����ʱ��
  for(int ImgWidth = 100; ImgWidth < depth_rows-100; ImgWidth++)
  {
    for(int ImgHeight = 200; ImgHeight < depth_cols-200; ImgHeight++ )
    {
      //��ȡ���ͼ�ж�Ӧ������ֵ
      float d = cv_ptr->image.at<float>(ImgWidth,ImgHeight);
      //�ж�mask���Ƿ�������ĵ�
      if(mask_ptr != 0)
      {
        unsigned char t = mask_ptr->image.at<unsigned char>(ImgWidth,ImgHeight);
        if(t == 0)
        continue;
      }
      else
      {
        if(d > 0.6)//�������ԣ�ͨ������ָ�����
        continue;
      }
      //���������Ŀռ�����
      pcl::PointXYZ PointWorld;
      PointWorld.z = double(d)/camera_factor;
      PointWorld.x = (ImgHeight - camera_cx)*PointWorld.z/camera_fx;
      PointWorld.y = (ImgWidth - camera_cy)*PointWorld.z/camera_fy;
      CloudMask->points.push_back(PointWorld);
    }
  }
  //���õ������ԣ������������з�ʽ�洢����
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
 
  //����label��ȡ����ģ��
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
    showCloudsLeft(CloudMask, CloudModel); //�����������򵥵���ʾԴ���ƺ�Ŀ�����
  }     
  //��׼����ģ�ͺͳ������������
  {
    pcl::ScopeTime scope_time("*PrePairAlign");//�����㷨����ʱ��
    prePairAlign(CloudMask,CloudModel,CloudPreProcess,true);
  }
  if(true == DEBUG_VISUALIZER)
  {
    PCL_INFO ("Press q to continue.\n");
    p->spin();
  }
  {
    pcl::ScopeTime scope_time("*PairAlign");//�����㷨����ʱ��
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

//depthͼ��ʾ�Ļص�����    
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
  // //���ӻ����ͼ
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
  euler_Angle = Rotation_matrix.eulerAngles(2, 1, 0);//˳��Z, Y, X
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

  

//****************  ������  ************************
int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pcl_registration");
  ros::NodeHandle nh;

  // Rviz������̬���ӻ�timer
  ros::Timer timer = nh.createTimer(ros::Duration(0.1), Pose_Visualer);
  sensor_msgs::PointCloud2 output;
  pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("pcl_output", 1);//���������⣨topic��

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
    //����һ�� PCLVisualizer ����
    p = new pcl::visualization::PCLVisualizer (argc, argv, "Pairwise Registration"); //p��ȫ�ֱ���
    p->createViewPort (0.0, 0, 0.5, 1.0, vp_1); //����������
    p->createViewPort (0.5, 0, 1.0, 1.0, vp_2); //����������
    //cv::namedWindow("depth image");
  }
  //��������ָ��ͱ任����
  PointCloud::Ptr result (new PointCloud), source, pretarget(new PointCloud), target; //����3������ָ�룬�ֱ����ڽ����Դ���ƺ�Ŀ�����
  Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity (), pairTransform; 

  //ROS���������ڵ㹲ͬ���ɲ���
  if(true == DEBUG_WITH_OTHERS)
  {
    //�������ͼ
    depth_sub = nh.subscribe("/camera/depth_registered/sw_registered/image_rect", 1 , depthCb); 
    //����mask-RCNN������Ϣ
    MaskRCNN_sub = nh.subscribe("maskRCNN",1,MaskRCNNCB); 
    //cv::destroyWindow("depth image"); 
  }
  else 
  {
    //ģ��������ԣ���Ӳ�̶�ȡ���Ե��ƺ�ģ�ͣ�
    //��ʽΪdata[0]=���Ե���1
    //data[1]=����ģ��1
    //data[2]=���Ե���2
    //data[3]=�������2
    // ����
    std::vector<PCD, Eigen::aligned_allocator<PCD> > data; //ģ��
    loadData (argc, argv, data); //��ȡpcd�ļ����ݣ����������
    //����û�����
    if (data.empty ())
    {
      PCL_ERROR ("Syntax is: %s <source.pcd> <target.pcd> [*]", argv[0]); //�﷨
      PCL_ERROR ("[*] - multiple files can be added. The registration results of (i, i+1) will be registered against (i+2), etc"); //����ʹ�ö���ļ�
      return (-1);
    }
    PCL_INFO ("Loaded %d datasets.", (int)data.size ()); //��ʾ��ȡ�˶��ٸ������ļ�
      //�������еĵ����ļ�
    for (size_t i = 1; i < data.size (); ++i, ++i)
    {
      source = data[i-1].cloud; //Դ����
      target = data[i].cloud; //Ŀ�����
      showCloudsLeft(source, target); //�����������򵥵���ʾԴ���ƺ�Ŀ�����
      PointCloud::Ptr temp (new PointCloud); //������ʱ����ָ��
          //��ʾ������׼�ĵ����ļ����͸��Եĵ���
      PCL_INFO ("Aligning %s (%d points) with %s (%d points).\n", data[i-1].f_name.c_str (), source->points.size (), data[i].f_name.c_str (), target->points.size ());

      prePairAlign(source, target, pretarget, true);
      pairAlign (source, target, temp, pairTransform, true);
      //����ǰ��һ�Ե������ݣ��任��ȫ�ֱ任�С�
      pcl::transformPointCloud (*temp, *result, GlobalTransform);
      //����ȫ�ֱ任
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

