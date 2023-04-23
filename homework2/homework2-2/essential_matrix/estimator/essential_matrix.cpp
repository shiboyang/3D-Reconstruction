#include "essential_matrix.h"

#include <complex>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <iostream>

using namespace Eigen;

void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d> &points,
                                   std::vector<Eigen::Vector2d> *normed_points,
                                   Eigen::Matrix3d *matrix) {
    // Calculate centroid
    Eigen::Vector2d centroid(0, 0);
    for (const Eigen::Vector2d &point: points) {
        centroid += point;
    }
    centroid /= points.size();

    // Root mean square error to centroid of all points
    double rms_mean_dist = 0;
    for (const Eigen::Vector2d &point: points) {
        rms_mean_dist += (point - centroid).squaredNorm();
    }
    rms_mean_dist = std::sqrt(rms_mean_dist / points.size());

    // Compose normalization matrix
    const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
    *matrix << norm_factor, 0, -norm_factor * centroid(0),
            0, norm_factor, -norm_factor * centroid(1),
            0, 0, 1;

    // Apply normalization matrix
    normed_points->resize(points.size());

    const double M_00 = (*matrix)(0, 0);
    const double M_01 = (*matrix)(0, 1);
    const double M_02 = (*matrix)(0, 2);
    const double M_10 = (*matrix)(1, 0);
    const double M_11 = (*matrix)(1, 1);
    const double M_12 = (*matrix)(1, 2);
    const double M_20 = (*matrix)(2, 0);
    const double M_21 = (*matrix)(2, 1);
    const double M_22 = (*matrix)(2, 2);

    for (size_t i = 0; i < points.size(); ++i) {
        const double p_0 = points[i](0);
        const double p_1 = points[i](1);

        const double np_0 = M_00 * p_0 + M_01 * p_1 + M_02;
        const double np_1 = M_10 * p_0 + M_11 * p_1 + M_12;
        const double np_2 = M_20 * p_0 + M_21 * p_1 + M_22;

        const double inv_np_2 = 1.0 / np_2;
        (*normed_points)[i](0) = np_0 * inv_np_2;
        (*normed_points)[i](1) = np_1 * inv_np_2;
        const Vector2d xy{np_0 * inv_np_2, np_1 * inv_np_2};
        std::cout << "center and normalized point dist: " << sqrt((xy - centroid).squaredNorm()) << "\n";
    }
}

Eigen::Matrix3d EssentialMatrixEightPointEstimate(const std::vector<Eigen::Vector2d> &points1,
                                                  const std::vector<Eigen::Vector2d> &points2) {

    // Center and normalize image points for better numerical stability.
    std::vector<Eigen::Vector2d> normed_points1;
    std::vector<Eigen::Vector2d> normed_points2;
    Eigen::Matrix3d points1_norm_matrix;
    Eigen::Matrix3d points2_norm_matrix;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

    Eigen::Matrix3d E;
    E.setIdentity();

    // homework4
    assert(points1.size() == points2.size() && points1.size() >= kMinNumSamples);
    Eigen::MatrixXd W(points1.size(), 9);
    Eigen::Matrix3d f2;
    Eigen::MatrixXd f1;
    double u, v, u1, v1;

    for (int i = 0; i < points1.size(); i++) {
        u = normed_points1[i](0);
        v = normed_points1[i](1);
        u1 = normed_points2[i](0);
        v1 = normed_points2[i](1);
        W.row(i) << u * u1, v * u1, u1, u * v1, v * v1, v1, u, v, 1;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd;
    svd.compute(W, Eigen::ComputeThinV | Eigen::ComputeThinU);
    // get latest column data
    f1 = svd.matrixV().col(svd.matrixV().cols() - 1);
    f1.resize(3, 3);
    f1.transposeInPlace();

    svd.compute(f1, Eigen::ComputeThinV | Eigen::ComputeThinU);
    auto S = svd.singularValues();
    S[2] = 0;
    f2 = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

    E = points2_norm_matrix.transpose() * f2 * points1_norm_matrix;
    return E;
}
//
//void center_and_normalize_point(const std::vector<Vector2d> &points, std::vector<Vector2d> &norm_points,
//                                Matrix3d &norm_matrix) {
//    Vector2d centroid{0, 0};
//    double mean_dist = 0;
//
//    for (const Vector2d &point: points) {
//        centroid += point;
//    }
//    centroid /= points.size();
//
//    for (const Vector2d &point: points) {
//        mean_dist += (point - centroid).squaredNorm();
//    }
//    mean_dist = sqrt(mean_dist / points.size());
//    const double factor = sqrt(2.0) / mean_dist;
//
//    for (const Vector2d &point: points) {
//        Vector2d xy;
//        xy << (point - centroid) * factor;
//        norm_points.push_back(xy);
//    }
//    norm_matrix << factor, 0, -factor * centroid(0),
//            0, factor, -factor * centroid(1),
//            0, 0, 1;
//}
//
//
//Matrix3d EssentialMatrixEightPointEstimate(const std::vector<Vector2d> &points1,
//                                           const std::vector<Vector2d> &points2) {
//
//    Matrix3d E, norm_matrix1, norm_matrix2;
//    std::vector<Vector2d> norm_points1, norm_points2;
//    MatrixXd W(points1.size(), 9), F1, F2;
//
//    center_and_normalize_point(points1, norm_points1, norm_matrix1);
//    center_and_normalize_point(points2, norm_points2, norm_matrix2);
//
//    for (int i = 0; i < norm_points1.size(); ++i) {
//        const double u = norm_points1[i](0);
//        const double v = norm_points1[i](1);
//        const double up = norm_points2[i](0);
//        const double vp = norm_points2[i](1);
//        W.row(i) << u * up, v * up, up, u * vp, v * vp, vp, u, v, 1;
//    }
//    JacobiSVD<MatrixXd> svd;
//    svd.compute(W, Eigen::ComputeThinV | Eigen::ComputeThinU);
//    F1 = svd.matrixV().col(svd.matrixV().cols() - 1);
//    F1.resize(3, 3);
//    F1.transposeInPlace();
//
//    svd.compute(F1, Eigen::ComputeThinV | Eigen::ComputeThinU);
//    auto S = svd.singularValues();
//    S[2] = 0;
//    F2 = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();
//
//    E = norm_matrix2.transpose() * F2 * norm_matrix1;
//    return E;
//}


Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Matrix3d &R,
                                        const Eigen::Vector3d &t) {
    Eigen::Matrix3d E;
    E.setIdentity();

    // homework3
    // Translate the t to skew-symmetric matrix for do t cross product R (T dot product R ==> t X R)
    Eigen::Matrix3d T;
    T << 0, -t(2), t(1),
            t(2), 0, -t(0),
            -t(1), t(0), 0;

    E = T * R;

    return E;
}

