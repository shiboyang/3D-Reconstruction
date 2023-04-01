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
    RowVectorXf x(12);
    x.setRandom();
    std::cout << "Initial x: " << x << std::endl;
    auto reshaped_x = x.reshaped(4, 3);
    cout << "Reshaped x:\n" << reshaped_x << "\n";
    cout << "Reshaped x.t:\n" << reshaped_x.transpose() << "\n";


    return 0;
}