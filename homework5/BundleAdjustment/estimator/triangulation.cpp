#include "triangulation.h"

Eigen::Vector3d TriangulatePoint(const Eigen::Matrix3x4d& proj_matrix1,
                                 const Eigen::Matrix3x4d& proj_matrix2,
                                 const Eigen::Vector2d& point1,
                                 const Eigen::Vector2d& point2) {
  Eigen::Matrix4d A;

  A.row(0) = point1(0) * proj_matrix1.row(2) - proj_matrix1.row(0);
  A.row(1) = point1(1) * proj_matrix1.row(2) - proj_matrix1.row(1);
  A.row(2) = point2(0) * proj_matrix2.row(2) - proj_matrix2.row(0);
  A.row(3) = point2(1) * proj_matrix2.row(2) - proj_matrix2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);

  return svd.matrixV().col(3).hnormalized();
}

std::vector<Eigen::Vector3d> TriangulatePoints(
    const Eigen::Matrix3x4d& proj_matrix1,
    const Eigen::Matrix3x4d& proj_matrix2,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) {
  
  std::vector<Eigen::Vector3d> points3D(points1.size());

  for (size_t i = 0; i < points3D.size(); ++i) {
    points3D[i] =
        TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
  }

  return points3D;
}

Eigen::Vector3d TriangulateMultiViewPoint(
    const std::vector<Eigen::Matrix3x4d>& proj_matrices,
    const std::vector<Eigen::Vector2d>& points) {
  
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

  for (size_t i = 0; i < points.size(); i++) {
    const Eigen::Vector3d point = points[i].homogeneous().normalized();
    const Eigen::Matrix3x4d term =
        proj_matrices[i] - point * point.transpose() * proj_matrices[i];
    A += term.transpose() * term;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(A);

  return eigen_solver.eigenvectors().col(0).hnormalized();
}
