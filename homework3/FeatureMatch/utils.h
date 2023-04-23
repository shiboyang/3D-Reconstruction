//
// Created by sparkai on 23-4-20.
//

#ifndef FEATUREMATCH_UTILS_H
#define FEATUREMATCH_UTILS_H

#include <iostream>
#include <iterator>
#include <vector>
#include <string>
#include <cmath>
#include <cfloat>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>


using namespace cv;
using namespace Eigen;

void translate_normal_camera_point(const std::vector<Point2f> &points,
                                   std::vector<Point2f> &norm_points,
                                   const Mat &K);

void extract_sift_feature(Mat &src,
                          std::vector<KeyPoint> &keypoints,
                          Mat &descriptors, int nfeature = 0);

void filterMatchedPoint(Mat &descriptors1,
                        Mat &descriptors2,
                        std::vector<DMatch> &match,
                        float thresh = 0.6);

void center_and_normalize_point(const std::vector<Vector2d> &points,
                                std::vector<Vector2d> &norm_points,
                                Matrix3d &norm_matrix);

Matrix3d estimate_essential_matrix(const std::vector<Vector2d> &points1,
                                   const std::vector<Vector2d> &points2);


std::vector<uint> sample_some_int(int first, int last, int k, RNG rng);

int calculate_inliers(const Matrix3d &E,
                      std::vector<Point2f> &pts1,
                      std::vector<Point2f> &pts2,
                      double t,
                      std::vector<uchar> &inliers);


void image_show(Mat &src,
                std::vector<KeyPoint> &keypoints);

void image_show(Mat &img1,
                std::vector<KeyPoint> &keypoints1,
                Mat &img2,
                std::vector<KeyPoint> &keypoints2,
                std::vector<DMatch> &matches,
                int numMatches = 10);


void image_show(Mat &img1, std::vector<Point2f> &pts1, Mat &img2, std::vector<Point2f> &pts2,
                std::vector<DMatch> &matches, int numMatches);

#endif //FEATUREMATCH_UTILS_H
