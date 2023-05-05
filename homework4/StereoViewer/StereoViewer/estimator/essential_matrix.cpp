#include "essential_matrix.h"

#include <complex>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "utils.h"


Eigen::Matrix3d EssentialMatrixEightPointEstimate(const std::vector<Eigen::Vector2d>& points1,
                                                  const std::vector<Eigen::Vector2d>& points2) {

    // Center and normalize image points for better numerical stability.
    std::vector<Eigen::Vector2d> normed_points1;
    std::vector<Eigen::Vector2d> normed_points2;
    Eigen::Matrix3d points1_norm_matrix;
    Eigen::Matrix3d points2_norm_matrix;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

    // Setup homogeneous linear equation as x2' * F * x1 = 0.
    Eigen::Matrix<double, Eigen::Dynamic, 9> cmatrix(points1.size(), 9);
    for (size_t i = 0; i < points1.size(); ++i) {
      cmatrix.block<1, 3>(i, 0) = normed_points1[i].homogeneous();
      cmatrix.block<1, 3>(i, 0) *= normed_points2[i].x();
      cmatrix.block<1, 3>(i, 3) = normed_points1[i].homogeneous();
      cmatrix.block<1, 3>(i, 3) *= normed_points2[i].y();
      cmatrix.block<1, 3>(i, 6) = normed_points1[i].homogeneous();
    }

    // Solve for the nullspace of the constraint matrix.
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> cmatrix_svd(
        cmatrix, Eigen::ComputeFullV);
    const Eigen::VectorXd ematrix_nullspace = cmatrix_svd.matrixV().col(8);
    const Eigen::Map<const Eigen::Matrix3d> ematrix_t(ematrix_nullspace.data());

    // De-normalize to image points.
    const Eigen::Matrix3d E_raw = points2_norm_matrix.transpose() *
                                  ematrix_t.transpose() * points1_norm_matrix;

    // Enforcing the internal constraint that two singular values must be equal
    // and one must be zero.
    Eigen::JacobiSVD<Eigen::Matrix3d> E_raw_svd(
        E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = E_raw_svd.singularValues();
    singular_values(0) = (singular_values(0) + singular_values(1)) / 2.0;
    singular_values(1) = singular_values(0);
    singular_values(2) = 0.0;
    const Eigen::Matrix3d E = E_raw_svd.matrixU() * singular_values.asDiagonal() *
                              E_raw_svd.matrixV().transpose();

    return E;
}

Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Matrix3d& R,
                                        const Eigen::Vector3d& t) {
    return CrossProductMatrix(t.normalized()) * R;
}

void DecomposeEssentialMatrix(const Eigen::Matrix3d& E, Eigen::Matrix3d* R1,
                              Eigen::Matrix3d* R2, Eigen::Vector3d* t) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV().transpose();

  if (U.determinant() < 0) {
    U *= -1;
  }
  if (V.determinant() < 0) {
    V *= -1;
  }

  Eigen::Matrix3d W;
  W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

  *R1 = U * W * V;
  *R2 = U * W.transpose() * V;
  *t = U.col(2).normalized();
}

void PoseFromEssentialMatrix(const Eigen::Matrix3d& E,
                             const std::vector<Eigen::Vector2d>& points1,
                             const std::vector<Eigen::Vector2d>& points2,
                             Eigen::Matrix3d* R, Eigen::Vector3d* t,
                             std::vector<Eigen::Vector3d>* points3D) {

  Eigen::Matrix3d R1;
  Eigen::Matrix3d R2;
  DecomposeEssentialMatrix(E, &R1, &R2, t);

  // Generate all possible projection matrix combinations.
  const std::array<Eigen::Matrix3d, 4> R_cmbs{{R1, R2, R1, R2}};
  const std::array<Eigen::Vector3d, 4> t_cmbs{{*t, *t, -*t, -*t}};

  points3D->clear();
  for (size_t i = 0; i < R_cmbs.size(); ++i) {
    std::vector<Eigen::Vector3d> points3D_cmb;
    CheckCheirality(R_cmbs[i], t_cmbs[i], points1, points2, &points3D_cmb);
    if (points3D_cmb.size() >= points3D->size()) {
      *R = R_cmbs[i];
      *t = t_cmbs[i];
      *points3D = points3D_cmb;
    }
  }
}