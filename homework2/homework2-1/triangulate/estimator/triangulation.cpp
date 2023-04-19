#include "triangulation.h"
#include <iostream>

Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d &proj_matrix1,
                                 const Eigen::Matrix3x4d &proj_matrix2,
                                 const Eigen::Vector2d &point1,
                                 const Eigen::Vector2d &point2) {
    // homework1
    Eigen::Vector3d point;
    Eigen::Matrix4d A;
    Eigen::Vector4d h_point;

    auto u = point1(0);
    auto v = point1(1);
    auto u1 = point2(0);
    auto v1 = point2(1);

    A << u * proj_matrix1.row(2) - proj_matrix1.row(0),
            v * proj_matrix1.row(2) - proj_matrix1.row(1),
            u1 * proj_matrix2.row(2) - proj_matrix2.row(0),
            v1 * proj_matrix2.row(2) - proj_matrix2.row(1);


    Eigen::JacobiSVD<Eigen::Matrix4d> svd;
    svd.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    h_point = svd.matrixV().col(svd.matrixV().cols() - 1);
    h_point /= h_point(3);
    point = h_point(Eigen::seq(0, 2));

    return point;

}

std::vector<Eigen::Vector3d> TriangulatePoints(
        const Eigen::Matrix3x4d &proj_matrix1,
        const Eigen::Matrix3x4d &proj_matrix2,
        const std::vector<Eigen::Vector2d> &points1,
        const std::vector<Eigen::Vector2d> &points2) {
    // homework2
    Eigen::Vector3d x;
    std::vector<Eigen::Vector3d> points3D(points1.size());
    for (int i = 0; i < points1.size(); ++i) {
        x = TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
        points3D.push_back(x);
    }

    return points3D;
}
