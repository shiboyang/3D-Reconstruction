//
// Created by shiby on 23-3-30.
//
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/SVD"
#include "iostream"
#include "cmath"

using namespace Eigen;
using namespace std;

using namespace Eigen;
using std::cout;

int main() {
    RowVectorXf x;
    RowVectorXf y;
    x.setRandom(1, 3);
    y = x * 2;

    cout << "x: " << x << "\n"
         << "y: " << y << "\n"
         << "y/x: " << y.array() / x.array();


    return 0;
}