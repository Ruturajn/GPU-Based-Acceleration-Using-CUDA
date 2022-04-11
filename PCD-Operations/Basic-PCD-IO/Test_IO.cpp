#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>


using namespace std;
using namespace pcl;

int main(void){
    
    PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);

    if (io::loadPLYFile<PointXYZ> ("/home/ruturajn/Downloads/Pandaset/Lidar/0001_test_2.ply", *cloud) == -1)
    {
        PCL_ERROR("Couldn't read the file scan_Velodyne_VLP16.pcd \n");
        return (-1);
    }

    cout << "Loaded "
         << cloud->width * cloud->height
         << " data points from scan_Velodyne_VLP16.pcd with the following fields: "
         << endl;
    
    // for (const auto& point: *cloud)
    //          cout << "    " << point.x
    //          << " "    << point.y
    //          << " "    << point.z << endl;

    return 0;
}

