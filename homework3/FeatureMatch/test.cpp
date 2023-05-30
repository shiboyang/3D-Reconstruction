//
// Created by sparkai on 23-4-20.
//

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <typeinfo>

using namespace std;
using namespace cv;
using namespace Eigen;

int main() {
    Eigen::Matrix3d matrix; // 定义一个3x3的矩阵
    Eigen::Vector3d vector; // 定义一个3x1的向量

// 将矩阵和向量的元素赋值为任意值
    matrix << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    vector << 1, 2, 3;


    std::cout << vector(2) << "\n";


    return 0;
}