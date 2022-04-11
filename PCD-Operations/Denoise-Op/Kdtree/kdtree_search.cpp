#include <iostream>
#include <iomanip>
#include <random>
#include <bits/stdc++.h>

// Define the k (Dimension) and total points
#define k 2
#define total_points 7

using namespace std;

// Define structure for a point
struct def_point
{
    int x;
    int y;
};

// Define structure for a node
struct parent_node
{
    def_point node_me;
    parent_node *node_left;
    parent_node *node_right;
};

// Function to insert a new point in the kdtree
parent_node *point_insert(def_point point_, parent_node *root_node, int depth)
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
    if ((point_.x == root_node->node_me.x) && (point_.y == root_node->node_me.y))
        return root_node;

    // Defining axis param to determine the axis to be considered
    // while splitting the plane
    int axis = depth % k;

    // Check the axis value, and perform splitting based on that
    // axis = 0 -> x-axis | axis = 1 -> y-axis
    if (axis == 0)
    {
        if (point_.x >= root_node->node_me.x)
            root_node->node_right = point_insert(point_, root_node->node_right, depth + 1);
        else
            root_node->node_left = point_insert(point_, root_node->node_left, depth + 1);
    }
    else
    {
        if (point_.y >= root_node->node_me.y)
            root_node->node_right = point_insert(point_, root_node->node_right, depth + 1);
        else
            root_node->node_left = point_insert(point_, root_node->node_left, depth + 1);
    }
    return root_node;
}

// Function to build a kdtree
parent_node *build_kdtree(parent_node *ktree_f, def_point *points_,
                          int num_points, int depth)
{
    for (int i = 0; i < num_points; i++)
        ktree_f = point_insert(points_[i], ktree_f, depth);
    return ktree_f;
}

bool search_node(parent_node *ktree_f, def_point point_, int depth)
{
    // Check if the root node is NULL
    if (ktree_f == NULL)
    {
        return false;
    }

    // Check if the point to be inserted is the root node itself
    if ((point_.x == ktree_f->node_me.x) && (point_.y == ktree_f->node_me.y))
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
    else
    {
        if (point_.y >= ktree_f->node_me.y)
            return search_node(ktree_f->node_right, point_, depth+1);
        else
            return search_node(ktree_f->node_left, point_, depth+1);
    }
    return search_node(ktree_f, point_, depth+1);
}

// Main function
int main()
{
    // Define array for the data_points
    def_point data_points[total_points] = {{3, 6}, {17, 15}, {13, 15}, {6, 12}, {9, 1}, {2, 7}, {10, 19}};

    // Print out the data points
    cout << "------------------------------- Data ------------------------------------------" << endl;

    for (int i = 0; i < total_points; i++)
    {
        // data_points[i].x = rand() % 100;
        // data_points[i].y = rand() % 100;
        cout << "(" << data_points[i].x << "," << data_points[i].y << ")" << endl;
    }
    cout << "-------------------------------------------------------------------------------" << endl;

    // Define the root node for the kdtree
    parent_node *ktree = NULL;
    ktree = build_kdtree(ktree, data_points, total_points, 0);

    // Search for a point
    def_point test_point = {10,10};
    bool search_res = search_node(ktree, test_point, 0);

    if (search_res == true)
        cout << "Found Point " << "(" << test_point.x << "," << test_point.y << ") : True" << endl;
    else
        cout << "Found Point " << "(" << test_point.x << "," << test_point.y << ") : False" << endl;
    return 0;
}
