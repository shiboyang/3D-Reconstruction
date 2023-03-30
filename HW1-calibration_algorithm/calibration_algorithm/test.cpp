//
// Created by shiby on 23-3-30.
//
#include "eigen3/Eigen/Dense"
#include "iostream"

using namespace Eigen;
using namespace std;

int main(int argc, char **argv) {

//    MatrixXf m = MatrixXf::Random(14, 12);
//    cout << "Here is the matrix m:" << endl << m << endl;
//    JacobiSVD<MatrixXf> svd(m);
//    svd.compute(m, ComputeThinU | ComputeThinV);
//
//    auto v = svd.matrixV();
//    std::cout << "v: " << v << "\nsize:" << v.cols() << "x" << v.rows() << std::endl;
//    std::cout << "last row norm: " << m * v.row(11).transpose() << std::endl;
//    std::cout << "last col norm: " << m * v.col(11) << std::endl;

    MatrixXf m = MatrixXf::Random(5, 3);
    std::cout << "m is \n" << m << std::endl;

    RowVector3f a(m.row(0).data());
    std::cout << "a is \n" << a << std::endl;

    return 0;
}