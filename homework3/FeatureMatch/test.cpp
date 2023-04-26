//
// Created by sparkai on 23-4-20.
//

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

int main() {
    Mat img;
    Matrix<int, 100, 100> m;
    m.fill(255);
    eigen2cv(m, img);
    return 0;
}