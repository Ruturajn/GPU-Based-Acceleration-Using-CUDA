/*
 * *************************************************************************** 
 * 
 * @file    main.cpp
 * @author  Ruturaj A. Nanoti
 * @brief   This program contains the main function for CUDA-based Point Cloud
 *          rotation and translation.
 * @date    14 March, 2022
 *
 * ***************************************************************************
*/

#include <iostream>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace pcl;

// Functions from Matrix_Transform.cu
int Mat_Transform(PointCloud<PointXYZ>::Ptr &cloud_in, PointCloud<PointXYZ>::Ptr &cloud_out);
void MatTransformKernel(PointXYZ *d_points, float *rot_matrix,
                        float *trans_matrix, int num_points);
void MatTransform(PointCloud<PointXYZ>::Ptr &cloud_in, PointCloud<PointXYZ>::Ptr &cloud_out);

int main(void)
{
    // Create a pointer of type PointCloud<PointXYZ>
    PointCloud<PointXYZ>::Ptr cloud_in(new PointCloud<PointXYZ>());
    PointCloud<PointXYZ>::Ptr cloud_out(new PointCloud<PointXYZ>());

    // Read the pcl file for the point cloud data
    io::loadPLYFile("../airplane.ply", *cloud_in);
    io::loadPLYFile("../airplane.ply", *cloud_out);
    // io::loadPLYFile("../teapot.ply", *cloud_in);
    // io::loadPLYFile("../teapot.ply", *cloud_out);
    //io::loadPCDFile("../denoised_teapot.pcd",*cloud_in);
    //io::loadPCDFile("../denoised_teapot.pcd",*cloud_out);

    // Call the pre-processing function
    Mat_Transform(cloud_in, cloud_out);

    return 0;
}
