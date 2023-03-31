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
    MatrixXd C;
    C.setRandom(27, 18);
    JacobiSVD<MatrixXd> svd(C, ComputeThinU | ComputeThinV);
    MatrixXd Cp = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV().transpose();
    MatrixXd diff = Cp - C;
    cout << "diff:\n" << diff.array().abs().sum() << "\n";

    MatrixXd v = svd.matrixV().rightCols(1);
    cout << "C @ v: " << (C * v) << "\n";



    return 0;
}