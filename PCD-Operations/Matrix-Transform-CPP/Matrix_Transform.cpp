/*
 * ****************************************************************************
 *
 * @file    Matrix_Transform.cu
 * @author  Ruturaj A. Nanoti
 * @brief   This program contains the GPU Kernel and the pre-processing function
 *          for point cloud rotation and translation.
 * @date    14 March, 2022
 *
 * ****************************************************************************
 */
#include <iostream>
#include <math.h>
#include <chrono>
#include <sys/time.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#define THREADSx 256
#define theta 45

using namespace std;
using namespace pcl;

void MatTransformFunc(PointXYZ *points_in, float *rot_matrix, float *trans_matrix,
                      int num_points, PointXYZ *points_out){
    
    for (int tid=0;tid<num_points;tid++){
        float prod_out[3] = {0};

        for (int i=0;i<3;i++){
            prod_out[i] += rot_matrix[i * 3] * points_in[tid].x;
            prod_out[i] += rot_matrix[i * 3 + 1] * points_in[tid].y;
            prod_out[i] += rot_matrix[i * 3 + 2] * points_in[tid].z;
        }
        points_out[tid].x = prod_out[0] + trans_matrix[0];
        points_out[tid].y = prod_out[1] + trans_matrix[1];
        points_out[tid].z = prod_out[2] + trans_matrix[2];
    }
}

int Mat_Transform(PointCloud<PointXYZ>::Ptr &cloud_in, PointCloud<PointXYZ>::Ptr &cloud_out)
{   
    
    struct timeval t1, t2;
    // Printing the number of points loaded
    cout << "Loaded " << cloud_in->width * cloud_in->height
         << " data points" << endl;

    float rot_matrix[9] = {cos(theta), -sin(theta), 0.0,
                           sin(theta), cos(theta), 0.0,
                           0.0, 0.0, 1.0};

    float trans_matrix[3] = {2.5, 0.0, 0.0};

    // Declare Device variables
    //auto start_cpu = std::chrono::high_resolution_clock::now();

    PointXYZ *points_in, *points_out;

    points_in = (PointXYZ *)malloc(cloud_in->points.size() * sizeof(PointXYZ));
    points_out = (PointXYZ *)malloc(cloud_in->points.size() * sizeof(PointXYZ));

    memcpy(points_in, cloud_in->points.data(),  cloud_in->points.size() * sizeof(PointXYZ));

    gettimeofday(&t1,0);
    MatTransformFunc(points_in, rot_matrix, trans_matrix, 
                     cloud_in->points.size(),points_out);
    gettimeofday(&t2,0);
    double time = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000.0; // Time taken by kernel in seconds
    //auto diff = std::chrono::high_resolution_clock::now() - start_cpu;
    //auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(diff);

    cout << "Time Required to Perform Point Cloud Transformation : " << time << endl;

    memcpy(cloud_out->points.data(), points_out, cloud_in->points.size() * sizeof(PointXYZ));

    // Visualizing the result
    cout << "Visualizing Point Clouds ..." << endl;
    cout << "White : Original Point Cloud " << endl;
    cout << "Red : Transformed Point Cloud " << endl;

    // Intializing the view window
    visualization::PCLVisualizer viewer("Transform operation Example");

    // Assigning colour to the original point clous and adding it to the viewer
    visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_in_handler(cloud_in, 255, 255, 255);
    viewer.addPointCloud(cloud_in, cloud_in_handler, "Original_Point_Cloud");

    // Assigning colour to the original point clous and adding it to the viewer
    visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_out_handler(cloud_out, 230, 20, 20);
    viewer.addPointCloud(cloud_out, cloud_out_handler, "Transformed_Point_Cloud");

    // Rendering the Point Cloud
    viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Original_Point_Cloud");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Transformed_Point_Cloud");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }
    return 0;
}