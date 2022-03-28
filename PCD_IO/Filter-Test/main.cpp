#include <iostream>
#include <iomanip>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>


using namespace std;
using namespace pcl;

#define THREADSx 256

void FilterCloudKernel(PointXYZ *d_points, float threshold, int num_points);
int FilterFunc(PointCloud<PointXYZ>::Ptr &cloud_in, PointCloud<PointXYZ>::Ptr &cloud_out);

int main(void){

    // Create a pointer of type PointCloud<PointXYZ>
    PointCloud<PointXYZ>::Ptr cloud_in (new PointCloud<PointXYZ>());
    PointCloud<PointXYZ>::Ptr cloud_out (new PointCloud<PointXYZ>());

    // Declare a blob to which the pcd file will write
    // This data will later be transferred to the PointCloud Pointer
    // PCLPointCloud2 cloud_blob;
    // io::loadPLYFile("./airplane.ply", cloud_blob);
    // fromPCLPointCloud2(cloud_blob, *cloud_in);
    // fromPCLPointCloud2(cloud_blob, *cloud_out);

    io::loadPLYFile("../airplane.ply", *cloud_in);
    io::loadPLYFile("../airplane.ply", *cloud_out);

    FilterFunc(cloud_in, cloud_out);

    return 0;
}