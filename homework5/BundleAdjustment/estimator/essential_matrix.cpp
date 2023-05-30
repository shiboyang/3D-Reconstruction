#include "essential_matrix.h"
#include "triangulation.h"
#include <complex>
#include <iostream>

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include "utils.h"


Eigen::Matrix3d EssentialMatrixEightPointEstimate(const std::vector<Eigen::Vector2d> &points1,
                                                  const std::vector<Eigen::Vector2d> &points2) {

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

Eigen::Matrix3d EssentialMatrixFromPose(const Eigen::Matrix3d &R,
                                        const Eigen::Vector3d &t) {
    return CrossProductMatrix(t.normalized()) * R;
}

void DecomposeEssentialMatrix(const Eigen::Matrix3d &E, Eigen::Matrix3d *R1,
                              Eigen::Matrix3d *R2, Eigen::Vector3d *t) {
    //////////////////// homework1 ////////////////
    //分解本质矩阵E -> R T
    // 构造矩阵 W Z
    const Eigen::Matrix3d W{{0, -1, 0},
                            {1, 0,  0},
                            {0, 0,  1}};
    const Eigen::Matrix3d Z{{0,  1, 0},
                            {-1, 0, 0},
                            {0,  0, 0}};
    Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E, Eigen::ComputeFullV | Eigen::ComputeFullU);
    const Eigen::Matrix3d &U = svd_E.matrixU();
    const Eigen::Vector3d &D = svd_E.singularValues();
    const Eigen::Matrix3d &V = svd_E.matrixV();

//    std::cout << "E\n" << E << "\n";
//    std::cout << "U*D*Vt\n" << U * D.asDiagonal() * Vt.transpose() << "\n";

    // If Z = diag(1,1,0) * W
    *R1 = U * W.transpose() * V.transpose();
    // 因为R为旋转矩阵 旋转矩阵的需要求行列式的值为正值
    *R1 = R1->determinant() * (*R1);
    // If Z = diag(1,1,0) * W.T
    *R2 = U * W * V.transpose();
    *R2 = R2->determinant() * (*R2);
    // T = U.col(2)
    *t = U.col(2);

    ///////////////////////////////////////////////
}


void PoseFromEssentialMatrix(const Eigen::Matrix3d &E, const Eigen::Matrix3d &K,
                             const std::vector<Eigen::Vector2d> &points1,
                             const std::vector<Eigen::Vector2d> &points2,
                             Eigen::Matrix3d *R, Eigen::Vector3d *t,
                             std::vector<Eigen::Vector3d> *points3D) {

    Eigen::Matrix3d R1;
    Eigen::Matrix3d R2;
    DecomposeEssentialMatrix(E, &R1, &R2, t);

    //////////////////// homework1 ////////////////

    // 根据DecomposeEssentialMatrix求出的R和t，还有图像中的points1和points2，求解两张图像的相对位姿
    // 可以参考opencv的modules/calib3d/src/five-point.cpp的recoverPose函数
    std::vector<Eigen::Vector3d> res1_points3D, res2_points3D, res3_points3D, res4_points3D;
    Eigen::Matrix3x4d P0, P1, P2, P3, P4;
    Eigen::MatrixXi mask(4, points1.size());
    mask.setZero();
    P0.setZero();
    P0.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    P0 = K * P0;
    P1 << R1, *t;
    P1 = K * P1;
    P2 << R1, -(*t);
    P2 = K * P2;
    P3 << R2, *t;
    P3 = K * P3;
    P4 << R2, -(*t);
    P4 = K * P4;
    res1_points3D = TriangulatePoints(P0, P1, points1, points2);
    for (int i = 0; i < res1_points3D.size(); ++i) {
        const Eigen::Vector3d Q1 = res1_points3D[i];
        const Eigen::Vector3d Q2 = P1 * Q1.homogeneous();
        if (Q1(2) > 0 && Q2(2) > 0) mask(0, i) = 1;
    }

    res2_points3D = TriangulatePoints(P0, P2, points1, points2);
    for (int i = 0; i < res2_points3D.size(); ++i) {
        const Eigen::Vector3d Q1 = res2_points3D[i];
        const Eigen::Vector3d Q2 = P2 * Q1.homogeneous();
        if (Q1(2) > 0 && Q2(2) > 0) mask(1, i) = 1;
    }

    res3_points3D = TriangulatePoints(P0, P3, points1, points2);
    for (int i = 0; i < res3_points3D.size(); ++i) {
        const Eigen::Vector3d Q1 = res3_points3D[i];
        const Eigen::Vector3d Q2 = P3 * Q1.homogeneous();
        if (Q1(2) > 0 && Q2(2) > 0) mask(2, i) = 1;
    }

    res4_points3D = TriangulatePoints(P0, P4, points1, points2);
    for (int i = 0; i < res4_points3D.size(); ++i) {
        const Eigen::Vector3d Q1 = res4_points3D[i];
        const Eigen::Vector3d Q2 = P4 * Q1.homogeneous();
        if (Q1(2) > 0 && Q2(2) > 0) mask(3, i) = 1;
    }

    int good1 = mask.row(0).sum();
    int good2 = mask.row(1).sum();
    int good3 = mask.row(2).sum();
    int good4 = mask.row(3).sum();

    if ((good1 > good2) && (good1 > good3) && (good1 > good4)) {
        *points3D = res1_points3D;
        *R = R1;
        *t = *t;
    } else if ((good2 > good1) && (good2 > good3) && (good2 > good4)) {
        *points3D = res2_points3D;
        *R = R1;
        *t = -(*t);
    } else if ((good3 > good1) && (good3 > good2) && (good3 > good4)) {
        *points3D = res3_points3D;
        *R = R2;
        *t = *t;
    } else {
        *points3D = res4_points3D;
        *R = R2;
        *t = -(*t);
    }

    ///////////////////////////////////////////////
}
