#include "epipolar_search.h"
#include <limits.h>
#include <iostream>

std::vector<cv::Point2f> convertPointsOfEigenToCV(const std::vector<Eigen::Vector2d> &points) {
    std::vector<cv::Point2f> points_cv;
    for(int i = 0; i < points.size(); i++) {
        points_cv.push_back(cv::Point2f(points[i](0), points[i](1)));
    }
    return points_cv;
}


typedef unsigned char uchar;

bool epipolarSearch(const cv::Mat &ref, const cv::Mat &curr, cv::Mat &disparity) {
    unsigned int numRows = ref.rows;
    unsigned int numCols = ref.cols;

    disparity = cv::Mat::zeros(numRows, numCols, CV_32F);

    for (int y = 0; y < numRows; y++) {

        int xR = 0;

        for (int x = 0; x < numCols; x++) {

            if (ref.at<uchar>(y, x) < 130.0 || ref.at<uchar>(y, x) > 254.0) continue;

            // Search for the corresponding pixel in the right image along the epipolar line
            int minDisparity = 64;
            int bestMatchX = -1;
            for (; xR < numCols; xR++) {

                if (curr.at<uchar>(y, xR) < 130.0 || curr.at<uchar>(y, xR) > 254.0) continue;

                std::cout << "ref.at<uchar>(y, x) "<<int (ref.at<uchar>(y, x))<<std::endl;

                int disparity = std::abs(ref.at<uchar>(y, x) - curr.at<uchar>(y, xR));
                if (disparity < minDisparity) {
                    minDisparity = disparity;
                    bestMatchX = xR;
                    break;
                }
            }
            // xR +;
            if(bestMatchX > numCols - 5) continue;
            if(bestMatchX < 0) continue;
            disparity.at<float>(y, x) = std::abs(x - bestMatchX);

            std::cout << "x is "<<x<<std::endl;
            std::cout << "y is "<<y<<std::endl;
            std::cout << "bestMatchX is "<<bestMatchX<<std::endl;

            std::cout <<" disparity "<<disparity.at<float>(y, x)<<std::endl;
        }
    }

    return true;
}

bool depthMapConverter(const cv::Mat &disparity, const double f, const double B, cv::Mat &depth) {
    unsigned int numRows = disparity.rows;
    unsigned int numCols = disparity.cols;

    depth = cv::Mat::zeros(numRows, numCols, CV_32F);

    for (int y = 0; y < numRows; y++) {
        for (int x = 0; x < numCols; x++) {
            if (disparity.at<float>(y, x) < 1.0) {
                depth.at<float>(y, x) = -1;
                continue;
            }

            depth.at<float>(y, x) = B * f / disparity.at<float>(y, x);
        }
    }

    return true;
}


std::vector<Eigen::Vector3i> DisparityPointsCalculate(std::vector<cv::Point2f> &points1, 
                                                      std::vector<cv::Point2f> &points2) {
    // homework3：平行视图的对应点，求出对应的视差。



}

std::vector<Eigen::Vector3d> Estimate3DPoints(std::vector<Eigen::Vector3i> &disparity,
                                              Eigen::Matrix4d &Twc,  Eigen::Matrix3d &K, 
                                              Eigen::Matrix3d H, double baseline ) {
    // homework4：求解二维点在三维空间中对应的位置

    
}