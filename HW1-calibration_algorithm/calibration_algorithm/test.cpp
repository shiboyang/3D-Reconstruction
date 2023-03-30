//
// Created by shiby on 23-3-30.
//
#include "eigen3/Eigen/Dense"
#include "iostream"


int main(int argc, char **argv) {

    Eigen::MatrixXf a(3, 3);
    a << 1, 2, 3, 4, 5, 6, 7, 8, 9;


//    a << 2;
//    a << 3;


    std::cout << a << std::endl;

//    std::cout << "the first row:" << a.row(1) << std::endl;
//    std::cout << "the second colum:" << a.col(2) << std::endl;
//    a.transposeInPlace();
//    std::cout << "transpose: " << a << std::endl;
//
//    a.resize(1, 9);
//    std::cout << a << std::endl;

    Eigen::RowVectorXf row(6);
    row << 1, 2, 2, 2, 2, 22;
    std::cout << row << std::endl;

    using namespace Eigen;
    using namespace std;

    MatrixXf m = MatrixXf::Random(3, 2);
    cout << "Here is the matrix m:" << endl << m << endl;
    JacobiSVD<MatrixXf> svd(m);
    svd.compute(m, ComputeThinU | ComputeThinV);
    cout << "Its singular values are:" << endl << svd.singularValues() << endl;
    cout << "Its left singular vectors are the columns of the thin U matrix:" << endl << svd.matrixU() << endl;
    cout << "Its right singular vectors are the columns of the thin V matrix:" << endl << svd.matrixV() << endl;
    Vector3f rhs(1, 0, 0);
    cout << "Now consider this rhs vector:" << endl << rhs << endl;
    cout << "A least-squares solution of m*x = rhs is:" << endl << svd.solve(rhs) << endl;


    return 0;
}