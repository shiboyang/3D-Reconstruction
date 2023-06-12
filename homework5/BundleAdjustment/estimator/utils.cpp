#include "utils.h"


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
        rms_mean_dist += (point - centroid).norm();
    }
    rms_mean_dist = rms_mean_dist / points.size();

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
        // const double np_2 = M_20 * p_0 + M_21 * p_1 + M_22;

        // const double inv_np_2 = 1.0 / np_2;
        // (*normed_points)[i](0) = np_0 * inv_np_2;
        // (*normed_points)[i](1) = np_1 * inv_np_2;
        (*normed_points)[i](0) = np_0;
        (*normed_points)[i](1) = np_1;

    }
}

Eigen::Matrix3d CrossProductMatrix(const Eigen::Vector3d &vector) {
    Eigen::Matrix3d matrix;
    matrix << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1),
            vector(0), 0;
    return matrix;
}

double CalculateDepth(const Eigen::Matrix3x4d &proj_matrix,
                      const Eigen::Vector3d &point3D) {
    const double proj_z = proj_matrix.row(2).dot(point3D.homogeneous());
    return proj_z * proj_matrix.col(2).norm();
}

Eigen::Matrix3x4d ComposeProjectionMatrix(const Eigen::Matrix3d &R,
                                          const Eigen::Vector3d &T) {
    Eigen::Matrix3x4d proj_matrix;
    proj_matrix.leftCols<3>() = R;
    proj_matrix.rightCols<1>() = T;
    return proj_matrix;
}

bool CheckCheirality(const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                     const std::vector<Eigen::Vector2d> &points1,
                     const std::vector<Eigen::Vector2d> &points2,
                     std::vector<Eigen::Vector3d> *points3D) {
    const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
    const Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(R, t);
    const double kMinDepth = std::numeric_limits<double>::epsilon();
    const double max_depth = 1000.0f * (R.transpose() * t).norm();
    points3D->clear();
    for (size_t i = 0; i < points1.size(); ++i) {
        const Eigen::Vector3d point3D =
                TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
        const double depth1 = CalculateDepth(proj_matrix1, point3D);
        if (depth1 > kMinDepth && depth1 < max_depth) {
            const double depth2 = CalculateDepth(proj_matrix2, point3D);
            if (depth2 > kMinDepth && depth2 < max_depth) {
                points3D->push_back(point3D);
            }
        }
    }
    return !points3D->empty();
}

Eigen::Matrix3x4d GetProjectMatrix(const Eigen::Matrix3d &intrinsic_matrix, Eigen::Matrix4d Tcw) {
    Eigen::Matrix3x4d ProjectMat;
    ProjectMat.setZero();
    ProjectMat.block<3, 3>(0, 0) = intrinsic_matrix;
    ProjectMat = ProjectMat * Tcw;
    return ProjectMat;
}