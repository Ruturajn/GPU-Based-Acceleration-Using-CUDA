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
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define THREADSx 256
#define theta 45

using namespace std;
using namespace pcl;

__global__ void MatTransformKernel(PointXYZ *d_points_in, float *rot_matrix,
                                   float *trans_matrix, int num_points, PointXYZ *d_points_out)
{

    // Defining global thread ID
    int tid = threadIdx.x + (blockDim.x * blockIdx.x);

    float prod_out[3];

    if (tid < num_points)
    {
        float prod = 0.0f;
        for (int i = 0; i < 3; i++)
        {
            prod_out[i] += rot_matrix[i * 3] * d_points_in[tid].x;
            prod_out[i] += rot_matrix[i * 3 + 1] * d_points_in[tid].y;
            prod_out[i] += rot_matrix[i * 3 + 2] * d_points_in[tid].z;
        }
        d_points_out[tid].x = prod_out[0] + trans_matrix[0];
        d_points_out[tid].y = prod_out[1] + trans_matrix[1];
        d_points_out[tid].z = prod_out[2] + trans_matrix[2];
    }
}

int Mat_Transform(PointCloud<PointXYZ>::Ptr &cloud_in, PointCloud<PointXYZ>::Ptr &cloud_out)
{

    // Printing the number of points loaded
    cout << "Loaded " << cloud_in->width * cloud_out->height
         << " data points from airplane.ply" << endl;

    float rot_matrix[9] = {cos(theta), -sin(theta), 0.0,
                           sin(theta), cos(theta), 0.0,
                           0.0, 0.0, 1.0};

    float trans_matrix[3] = {2.5, 0.0, 0.0};

    // Declare Device variables
    PointXYZ *d_points_in, *d_points_out;
    float *d_rot_matrix, *d_trans_matrix;

    // Allocating memory for points and matrices on the device
    cudaMalloc((void **)&d_points_in, cloud_in->points.size() * sizeof(PointXYZ));
    cudaMalloc((void **)&d_points_out, cloud_in->points.size() * sizeof(PointXYZ));
    cudaMalloc((void **)&d_rot_matrix, 9 * sizeof(float));
    cudaMalloc((void **)&d_trans_matrix, 3 * sizeof(float));

    // Copy the point and matrix data from host to device
    cudaMemcpy(d_points_in, cloud_in->points.data(), cloud_in->points.size() * sizeof(PointXYZ), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rot_matrix, rot_matrix, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_trans_matrix, trans_matrix, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate the threads and block required
    int BLOCKSx = (cloud_in->points.size() + THREADSx - 1) / THREADSx;

    dim3 threads(THREADSx);
    dim3 blocks(BLOCKSx);

    MatTransformKernel<<<blocks, threads>>>(d_points_in, d_rot_matrix, d_trans_matrix, cloud_in->points.size(), d_points_out);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Copy data back to host from device
    cudaMemcpy(cloud_out->points.data(), d_points_out, cloud_in->points.size() * sizeof(PointXYZ), cudaMemcpyDeviceToHost);

    // for (const auto &point : *cloud_out)
    //     cout << "    " << point.x
    //          << " " << point.y
    //          << " " << point.z << endl;

    // Free allocated memory
    cudaFree(d_points_in);
    cudaFree(d_points_out);
    cudaFree(d_rot_matrix);
    cudaFree(d_trans_matrix);

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