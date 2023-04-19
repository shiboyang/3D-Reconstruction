//
// Created by sparkai on 23-4-18.
//
#include <iostream>
#include "eigen3/Eigen/Core"


using namespace Eigen;
using namespace std;

int main(int argc, char **argv) {
    Vector2d p1{1, 2};
    auto p = p1.array() * p1.array();
    cout << p.sum() << endl;

    cout << p1.squaredNorm() << endl;

    return 0;
}