#pragma once

#include <vector>

#include <Eigen/Core>

#include "types.h"

#include "triangulation.h"

// Center and normalize image points.
//
// The points are transformed in a two-step procedure that is expressed
// as a transformation matrix. The matrix of the resulting points is usually
// better conditioned than the matrix of the original points.
//
// Center the image points, such that the new coordinate system has its
// origin at the centroid of the image points.
//
// Normalize the image points, such that the mean distance from the points
// to the coordinate system is sqrt(2).
//
// @param points          Image coordinates.
// @param normed_points   Transformed image coordinates.
// @param matrix          3x3 transformation matrix.
void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d>& points,
                                   std::vector<Eigen::Vector2d>* normed_points,
                                   Eigen::Matrix3d* matrix);

// Compose the skew symmetric cross product matrix from a vector.
Eigen::Matrix3d CrossProductMatrix(const Eigen::Vector3d& vector);

// Calculate depth of 3D point with respect to camera.
//
// The depth is defined as the Euclidean distance of a 3D point from the
// camera and is positive if the 3D point is in front and negative if
// behind of the camera.
//
// @param proj_matrix     3x4 projection matrix.
// @param point3D         3D point as 3x1 vector.
//
// @return                Depth of 3D point.
double CalculateDepth(const Eigen::Matrix3x4d& proj_matrix,
                      const Eigen::Vector3d& point3D);

// Compose projection matrix from rotation matrix and translation components).
//
// The projection matrix transforms 3D world to image points.
//
// @param R              3x3 rotation matrix.
// @param t              3x1 translation vector.
//
// @return               3x4 projection matrix.
Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Matrix3d& R,
                                          const Eigen::Vector3d& T);

// Perform cheirality constraint test, i.e., determine which of the triangulated
// correspondences lie in front of of both cameras. The first camera has the
// projection matrix P1 = [I | 0] and the second camera has the projection
// matrix P2 = [R | t].
//
// @param R            3x3 rotation matrix of second projection matrix.
// @param t            3x1 translation vector of second projection matrix.
// @param points1      First set of corresponding points.
// @param points2      Second set of corresponding points.
// @param points3D     Points that lie in front of both cameras.
bool CheckCheirality(const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                     const std::vector<Eigen::Vector2d>& points1,
                     const std::vector<Eigen::Vector2d>& points2,
                     std::vector<Eigen::Vector3d>* points3D);

Eigen::Matrix3x4d GetProjectMatrix(Eigen::Matrix3d intrinsic_matrix, Eigen::Matrix4d Tcw);