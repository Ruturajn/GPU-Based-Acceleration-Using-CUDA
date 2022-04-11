#include <iostream>
#include <iomanip>
#include <random>
#include <bits/stdc++.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// Define the k (Dimension)
#define k 3

using namespace std;
using namespace pcl;

// Define structure for a node
struct parent_node
{
    PointXYZ node_me;
    parent_node *node_left;
    parent_node *node_right;
};

// Function to insert a new point in the kdtree
parent_node *point_insert(PointXYZ point_, parent_node *root_node, int depth)
{
    // Check if the node is pointing to a null
    if (root_node == NULL)
    {
        parent_node *new_node = new parent_node;
        new_node->node_me = point_;
        new_node->node_left = new_node->node_right = NULL;
        return new_node;
    }

    // Check if the point to be inserted is the root node itself
    if ((point_.x == root_node->node_me.x) && (point_.y == root_node->node_me.y) && (point_.z == root_node->node_me.z))
        return root_node;

    // Defining axis param to determine the axis to be considered
    // while splitting the plane
    int axis = depth % k;

    // Check the axis value, and perform splitting based on that
    // axis = 0 -> x-axis | axis = 1 -> y-axis | axis = 2 -> z-axis 
    if (axis == 0)
    {
        if (point_.x >= root_node->node_me.x)
            root_node->node_right = point_insert(point_, root_node->node_right, depth + 1);
        else
            root_node->node_left = point_insert(point_, root_node->node_left, depth + 1);
    }
    else if (axis == 1)
    {
        if (point_.y >= root_node->node_me.y)
            root_node->node_right = point_insert(point_, root_node->node_right, depth + 1);
        else
            root_node->node_left = point_insert(point_, root_node->node_left, depth + 1);
    }
    else
    {
        if (point_.z >= root_node->node_me.z)
            root_node->node_right = point_insert(point_, root_node->node_right, depth + 1);
        else
            root_node->node_left = point_insert(point_, root_node->node_left, depth + 1);
    }
    return root_node;
}

// Function to build a kdtree
parent_node *build_kdtree(parent_node *ktree_f, PointXYZ *points_,
                          int num_points, int depth)
{
    for (int i = 0; i < num_points; i++)
        ktree_f = point_insert(points_[i], ktree_f, depth);
    return ktree_f;
}

bool search_node(parent_node *ktree_f, PointXYZ point_, int depth)
{
    // Check if the root node is NULL
    if (ktree_f == NULL)
    {
        return false;
    }

    // Check if the point to be inserted is the root node itself
    if ((point_.x == ktree_f->node_me.x) && (point_.y == ktree_f->node_me.y) && (point_.z == ktree_f->node_me.z))
        return true;

    // Defining axis param to determine the axis to be considered
    // while splitting the plane
    int axis = depth % k;

    // Check the axis value, and perform splitting based on that
    // axis = 0 -> x-axis | axis = 1 -> y-axis
    if (axis == 0)
    {
        if (point_.x >= ktree_f->node_me.x)
            return search_node(ktree_f->node_right, point_, depth+1);
        else
            return search_node(ktree_f->node_left, point_, depth+1);
    }
    else if (axis == 1)
    {
        if (point_.y >= ktree_f->node_me.y)
            return search_node(ktree_f->node_right, point_, depth+1);
        else
            return search_node(ktree_f->node_left, point_, depth+1);
    }
    else
    {
        if (point_.z >= ktree_f->node_me.z)
            return search_node(ktree_f->node_right, point_, depth+1);
        else
            return search_node(ktree_f->node_left, point_, depth+1);   
    }
    return search_node(ktree_f, point_, depth+1);
}

// Main function
int main()
{
    // Create a pointer of type PointCloud<PointXYZ>
    PointCloud<PointXYZ>::Ptr cloud_in(new PointCloud<PointXYZ>());

    //io::loadPLYFile("../../airplane.ply", *cloud_in);
    io::loadPCDFile("../../denoised_teapot.pcd", *cloud_in);

    PointXYZ *data_points;

    size_t cloud_size = cloud_in->points.size() * sizeof(PointXYZ);

    data_points = (PointXYZ *)malloc(cloud_size);
    memcpy(data_points, cloud_in->points.data(), cloud_size);

    // Define the root node for the kdtree
    parent_node *ktree = NULL;
    ktree = build_kdtree(ktree, data_points, cloud_in->points.size(), 0);

    // Search for a point
    //PointXYZ test_point = data_points[20000];
    PointXYZ test_point = {10,10,10};
    bool search_res = search_node(ktree, test_point, 0);

    if (search_res == true)
        cout << "Found Point " << "(" << test_point.x << "," << test_point.y << "," << test_point.z << ") : True" << endl;
    else
        cout << "Found Point " << "(" << test_point.x << "," << test_point.y << "," << test_point.z << ") : False" << endl;
    
    free(data_points);
    return 0;
}
