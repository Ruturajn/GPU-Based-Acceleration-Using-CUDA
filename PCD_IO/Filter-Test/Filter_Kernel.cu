#include <iostream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;
using namespace pcl;

#define THREADSx 256

__global__ void FilterCloudKernel(PointXYZ *d_points, float threshold, int num_points){
    
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    if (tid < num_points){
        //d_points[tid].x = (d_points[tid].x < threshold)? 0 : d_points[tid].x;
        d_points[tid].y = (d_points[tid].y < threshold)? 0 : d_points[tid].y;
        //d_points[tid].z = (d_points[tid].z < threshold)? 0 : d_points[tid].z;
    }
}

int FilterFunc(PointCloud<PointXYZ>::Ptr &cloud_in, PointCloud<PointXYZ>::Ptr &cloud_out){

    // // Create a pointer of type PointCloud<PointXYZ>
    // PointCloud<PointXYZ>::Ptr cloud_in (new PointCloud<PointXYZ>());
    // PointCloud<PointXYZ>::Ptr cloud_out (new PointCloud<PointXYZ>());

    // // Declare a blob to which the pcd file will write
    // // This data will later be transferred to the PointCloud Pointer
    // PCLPointCloud2 cloud_blob;
    // io::loadPLYFile("./airplane.ply", cloud_blob);
    // fromPCLPointCloud2(cloud_blob, *cloud_in);
    // fromPCLPointCloud2(cloud_blob, *cloud_out);

    // //io::loadPLYFile("./airplane.ply", *cloud_in);
    // //io::loadPLYFile("./airplane.ply", *cloud_out);

    // Print out the total number of points in the point cloud
    cout << "Loaded " << cloud_in->width * cloud_in->height
         << " data points from airplane.ply" << endl;

    //cout << cloud->points.data()[0] << endl;

    // PointXYZ *h_points;

    // h_points = (PointXYZ *)malloc(cloud->points.size()*sizeof(PointXYZ));
    // memcpy(h_points, cloud->points.data(), cloud->points.size()*sizeof(PointXYZ));

    // cout << h_points[0].x << endl;

    // free(h_points);
    
    
    // Declare device variables
    PointXYZ *d_points;

    // Allocating memory for the points on the device
    cudaMalloc((void**)&d_points, cloud_in->points.size()*sizeof(PointXYZ));

    // Copying points from host to device
    cudaMemcpy(d_points, cloud_in->points.data(), cloud_in->points.size()*sizeof(PointXYZ), cudaMemcpyHostToDevice);

    // Calculate the threads and blocks requrired
    int BLOCKSx = (cloud_in->points.size() + THREADSx - 1)/THREADSx;

    dim3 threads(THREADSx);
    dim3 blocks(BLOCKSx);

    FilterCloudKernel<<<blocks, threads>>>(d_points, 490.1f, cloud_in->points.size());
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Copy data() back to host
    cudaMemcpy(cloud_out->points.data(), d_points, cloud_in->points.size()*sizeof(PointXYZ), cudaMemcpyDeviceToHost);

    // for (const auto& point: *cloud_out)
    // cout << "    " << point.x
    //         << " "    << point.y
    //         << " "    << point.z << endl;

    // Free the allocated memory
    cudaFree(d_points);

    // Visualizing the result
    cout << "Visualizing Point Clouds ..." << endl;
    cout << "White : Original Point Cloud " << endl;
    cout << "Red : Threshold Operated Point Cloud " << endl;

    // Intializing the view window
    visualization::PCLVisualizer viewer("Thresholding operation Example");

    // Assigning colour to the original point clous and adding it to the viewer
    visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_in_handler(cloud_in, 255, 255, 255);
    viewer.addPointCloud(cloud_in, cloud_in_handler, "Original_Point_Cloud");

    // Assigning colour to the original point clous and adding it to the viewer
    visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_out_handler(cloud_out, 230, 20, 20);
    viewer.addPointCloud(cloud_out, cloud_out_handler, "Threshold_Operated_Point_Cloud");

    // Rendering the Point Cloud
    viewer.addCoordinateSystem(1.0, "cloud", 0);
    viewer.setBackgroundColor(0.05, 0.05, 0.05, 0);
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Original_Point_Cloud");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 2, "Threshold_Operated_Point_Cloud");

    while (!viewer.wasStopped()){
        viewer.spinOnce();
    }
    return 0;
}
