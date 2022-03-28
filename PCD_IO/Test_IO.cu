#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/cuda/point_cloud.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

using namespace std;
using namespace pcl;


int main(void){
    
    PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>());

    pcl::PCLPointCloud2 cloud_blob;
    pcl::io::loadPCDFile("/home/ruturajn/Documents/Project-Files/CUDA_Scripts/PCD_IO/scan_Velodyne_VLP16.pcd", cloud_blob);
    pcl::fromPCLPointCloud2 (cloud_blob, *cloud);

    
    // if (io::loadPCDFile<PointXYZ>("/home/ruturajn/Documents/Project-Files/CUDA_Scripts/PCD_IO/scan_Velodyne_VLP16.pcd", *cloud) == -1)
    // {
    //     PCL_ERROR("Couldn't read the file scan_Velodyne_VLP16.pcd \n");
    //     return (-1);
    // }


    cout << "Loaded "
         << cloud->width * cloud->height
         << " data points from scan_Velodyne_VLP16.pcd with the following fields: "
         << endl;
    
    for (const auto& point: *cloud)
        cout << "    " << point.x
             << " "    << point.y
             << " "    << point.z << endl;

    return 0;
}

