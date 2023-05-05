#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include "types.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


std::vector<cv::Point2f> convertPointsOfEigenToCV(const std::vector<Eigen::Vector2d> &points);

bool epipolarSearch(const cv::Mat &ref, const cv::Mat &curr, cv::Mat &disparity);

bool depthMapConverter(const cv::Mat &disparity, const double f, const double B, cv::Mat &depth);

std::vector<Eigen::Vector3i> DisparityPointsCalculate(std::vector<cv::Point2f> &points1, 
                                                      std::vector<cv::Point2f> &points2);

std::vector<Eigen::Vector3d> Estimate3DPoints(std::vector<Eigen::Vector3i> &disparity,
                                              Eigen::Matrix4d &Twc,  Eigen::Matrix3d &K, 
                                              Eigen::Matrix3d H, double baseline );