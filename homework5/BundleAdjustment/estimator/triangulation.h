#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "types.h"

// Triangulate 3D point from corresponding image point observations.
//
// Implementation of the direct linear transform triangulation method in
//   R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision,
//   Cambridge Univ. Press, 2003.
//
// @param proj_matrix1   Projection matrix of the first image as 3x4 matrix.
// @param proj_matrix2   Projection matrix of the second image as 3x4 matrix.
// @param point1         Corresponding 2D point in first image.
// @param point2         Corresponding 2D point in second image.
//
// @return               Triangulated 3D point.
Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d& proj_matrix1,
                                 const Eigen::Matrix3x4d& proj_matrix2,
                                 const Eigen::Vector2d& point1,
                                 const Eigen::Vector2d& point2);

// Triangulate multiple 3D points from multiple image correspondences.
std::vector<Eigen::Vector3d> TriangulatePoints(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2);

// Triangulate point from multiple views minimizing the L2 error.
//
// @param proj_matrices       Projection matrices of multi-view observations.
// @param points              Image observations of multi-view observations.
//
// @return                    Estimated 3D point.
Eigen::Vector3d TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix3x4d>& proj_matrices,
    const std::vector<Eigen::Vector2d>& points);
