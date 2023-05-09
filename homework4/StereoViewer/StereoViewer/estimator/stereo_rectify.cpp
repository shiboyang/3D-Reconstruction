#include <iostream>
#include <cmath>

#include "stereo_rectify.h"

void RectifyStereoCamerasByPoints(const Camera &camera,
                                  const std::vector<Eigen::Vector2d> &normal_points1,
                                  const std::vector<Eigen::Vector2d> &normal_points2,
                                  const std::vector<Eigen::Vector2d> &points1,
                                  const std::vector<Eigen::Vector2d> &points2,
                                  Eigen::Matrix3d *H1,
                                  Eigen::Matrix3d *H2) {

    // homework1：通过匹配的点进行双目立体视觉矫正的H矩阵计算
    Eigen::Matrix3d E, F, H, Hp;
    Eigen::Matrix3d K = camera.intrinsic_matrix;
    Eigen::Vector3d e, ep, translated_e, translated_ep;


    E = EssentialMatrixEightPointEstimate(normal_points1, normal_points2);

    F = K.inverse().transpose() * E * K.inverse();


    // 计算极点e, ep
    ep = calculate_epipolar(points1, F);
    e = calculate_epipolar(points2, F.transpose());

    //利用单应变换将坐标e点转换至无穷远点(f,0,0)  计算透视得到透视矩阵H
    *H2 = calculate_holograph_matrix2(ep, camera.image_w, camera.image_h);
    *H1 = calculate_holograph_matrix1(H2, F, ep, e, points1, points2);

    std::cout << "H1 * e\n" << H * e << "\n";
    std::cout << "H2 * e\n" << Hp * ep << "\n";
}


Eigen::Matrix3d calculate_holograph_matrix2(const Eigen::Vector3d &e,
                                            double w,
                                            double h) {

    // by translate the epipolar to infinite point
    Eigen::Matrix3d T, R, G, H;
    Eigen::Vector3d tmp_e;

    double alpha = 1;

    T << 1, 0, -w / 2,
            0, 1, -h / 2,
            0, 0, 1;

    tmp_e = T * e;

    if (tmp_e.x() < 0) alpha = -1;

    const double s = alpha / sqrt(tmp_e.x() * tmp_e.x() + tmp_e.y() * tmp_e.y());

    R << tmp_e.x() * s, tmp_e.y() * s, 0,
            -tmp_e.y() * s, tmp_e.x() * s, 0,
            0, 0, 1;

    tmp_e = R * tmp_e;
    G << 1, 0, 0,
            0, 1, 0,
            -1 / tmp_e.x(), 0, 1;

    H = T.inverse() * G * R * T;

    return H;
}


Eigen::Matrix3d calculate_holograph_matrix1(const Eigen::Matrix3d *H2,
                                            Eigen::Matrix3d &F,
                                            const Eigen::Vector3d &ep,
                                            const Eigen::Vector3d &e,
                                            const std::vector<Eigen::Vector2d> &pts1,
                                            const std::vector<Eigen::Vector2d> &pts2) {
    Eigen::Matrix3d H1;
    const int num_points = pts1.size();
    Eigen::Matrix3d Ha, M;

    M = CrossProductMatrix(ep) * F + (ep * Eigen::Vector3d::Ones().transpose());

    // Wa = b 求解a的值
    std::vector<Eigen::Vector3d> pts1_hat, pts2_hat;
    for (int i = 0; i < num_points; ++i) {
        const Eigen::Vector3d p1 = *H2 * M * Eigen::Vector3d{pts1[i].x(), pts1[i].y(), 1};
        const Eigen::Vector3d p2 = *H2 * Eigen::Vector3d{pts2[i].x(), pts2[i].y(), 1};

        pts1_hat.emplace_back(p1 / p1.z());
        pts2_hat.emplace_back(p2 / p2.z());
    }

    Eigen::MatrixXd W(num_points, 3);
    Eigen::VectorXd b(num_points);
    for (int i = 0; i < num_points; ++i) {
        const double x = pts1_hat[i].x();
        const double y = pts1_hat[i].y();
        const double xp = pts2_hat[i].x();
        W.row(i) << x, y, 1;
        b(i) = xp;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Vector3d singular_value = svd.singularValues();
    Eigen::MatrixXd U = svd.matrixU();
    const Eigen::Matrix3d &V = svd.matrixV();

    int c;
    for (double i: singular_value) {
        if (i > 1e-10) ++c;
    }
    assert(c >= W.cols());

    Eigen::VectorXd UTb = U.transpose() * b;
    Eigen::Vector3d a = V * singular_value.asDiagonal().inverse() * UTb.head(3);

    Ha << a(0), a(1), a(2),
            0, 1, 0,
            0, 0, 1;

    H1 = Ha * (*H2) * M;

    return H1;
}


Eigen::Vector3d calculate_epipolar(const std::vector<Eigen::Vector2d> &points1,
                                   const Eigen::Matrix3d &F) {
    Eigen::Vector3d epipolar;
    Eigen::MatrixXd L(points1.size(), 3);

    for (int i = 0; i < points1.size(); ++i) {
        L.row(i) = F * points1[i].homogeneous();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(L, Eigen::ComputeThinV | Eigen::ComputeThinU);

    epipolar = svd.matrixV().col(svd.matrixV().cols() - 1);

    return epipolar / epipolar.z();
}


void apply_holograph_matrix(const std::vector<Eigen::Vector2d> &points, const Eigen::Matrix3d &H) {
    Eigen::Vector3d tmp_p;
    for (const auto &p: points) {
        tmp_p = H * Eigen::Vector3d{p(0), p(1), 1};
        tmp_p /= tmp_p.z();
        std::cout << "(" << tmp_p.x() << ", " << tmp_p.y() << ")" << "\n";
    }
    std::cout << "------------" << "\n";
}