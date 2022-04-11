#include <iostream>
#include <iomanip>
#include <random>

#define k 2
#define total_points 15

using namespace std;

struct def_points
{
    int x;
    int y;
};

struct node
{
    def_points node_left;
    def_points node_right;
};

void build_kdtree(vector<node> &ktree_f, vector<def_points> &points_, int num_point_ind,
                  int axis, int num_points)
{
    if (num_point_ind > num_points - 1)
        return;
    else
    {
        if (axis == 0)
        {
            if (points_[num_point_ind].x > points_[num_point_ind + 1].x)
            {
                ktree_f[num_point_ind + 1].node_right.x = points_[num_point_ind + 1].x;
                ktree_f[num_point_ind + 1].node_right.y = points_[num_point_ind + 1].y;
            }
            else
            {
                ktree_f[num_point_ind + 1].node_left.x = points_[num_point_ind + 1].x;
                ktree_f[num_point_ind + 1].node_left.y = points_[num_point_ind + 1].y;
            }
            axis = 1;
        }
        else
        {
            if (points_[num_point_ind].y > points_[num_point_ind + 1].y)
            {
                ktree_f[num_point_ind + 1].node_right.y = points_[num_point_ind + 1].y;
                ktree_f[num_point_ind + 1].node_right.x = points_[num_point_ind + 1].x;
            }
            else
            {
                ktree_f[num_point_ind + 1].node_left.y = points_[num_point_ind + 1].y;
                ktree_f[num_point_ind + 1].node_left.x = points_[num_point_ind + 1].x;
            }
            axis = 0;
        }
        build_kdtree(ktree_f, points_, num_point_ind + 1, axis, num_points);
    }
}

int main()
{
    vector<node> ktree(total_points);
    vector<def_points> data_points(total_points);

    cout << "------------------------------- Data ------------------------------------------" << endl;

    for (int i = 0; i < 15; i++)
    {
        data_points[i].x = rand() % 100;
        data_points[i].y = rand() % 100;
        ktree[i].node_left.x = 0;
        ktree[i].node_left.y = 0;
        ktree[i].node_right.x = 0;
        ktree[i].node_right.y = 0;
        cout << "(" << data_points[i].x << "," << data_points[i].y << ")" << endl;
    }

    cout << "-------------------------------------------------------------------------------" << endl;

    int num_point_ind = 0;
    int axis = 0; // x -> 0 | y-> 1

    ktree[0].node_right.x = data_points[0].x;
    ktree[0].node_right.y = data_points[0].y;

    build_kdtree(ktree, data_points, num_point_ind, axis, total_points);
    for (int i = 0; i < total_points; i++)
    {
        cout << "Node : " << i << endl;
        cout << "(" << ktree[i].node_left.x << " , " << ktree[i].node_left.y << ") | (" << ktree[i].node_right.x << " , " << ktree[i].node_right.y << ")" << endl;
    }

    return 0;
}