#include "camera.h"
#include "essential_matrix.h"

#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/eigen.hpp>


int main(int argc, char **argv) {
    Camera camera = Camera(std::string(argv[1]));
    camera.GenerateFrames();

    Eigen::Matrix4d Twc1 = camera.GetFirstFrameTwc();
    Eigen::Matrix4d Twc2 = camera.GetSecondFrameTwc();

    std::vector<Eigen::Vector2d> points1 = camera.GetFirstFramePoints();
    std::vector<Eigen::Vector2d> points2 = camera.GetSecondFramePoints();

    std::vector<Eigen::Vector2d> normal_points1 = camera.GetFirstFrameNormalPoints();
    std::vector<Eigen::Vector2d> normal_points2 = camera.GetSecondFrameNormalPoints();

    Eigen::Matrix4d Tc1c2 = Twc1.inverse() * Twc2;
    Eigen::Matrix4d Tc2c1 = Twc2.inverse() * Twc1;

    Eigen::Matrix3d Rc2c1 = Tc2c1.block<3, 3>(0, 0);
    Eigen::Vector3d tc2c1 = Tc2c1.block<3, 1>(0, 3);

    double scale = tc2c1.norm();

    Eigen::Matrix3d E_gt = EssentialMatrixFromPose(Rc2c1, tc2c1);

    Eigen::Matrix3d E = EssentialMatrixEightPointEstimate(normal_points1, normal_points2);

    std::cout << "E gt " << std::endl << E_gt << std::endl;
    std::cout << std::endl;
    std::cout << "E  " << std::endl << E << std::endl;

    std::cout << "E_gt / E \n" << E_gt.array() / E.array() << std::endl;

    // verify E
    for (int i = 0; i < points1.size(); i++) {
        const Eigen::Vector3d p1(normal_points1[i][0], normal_points1[i][1], 1);
        const Eigen::Vector3d p2(normal_points2[i][0], normal_points2[i][1], 1);
        std::cout << i << "th point p'.T x E x p = " << p2.transpose() * E * p1 << std::endl;
        std::cout << i << "th point p'.T x E_gt x p = " << p2.transpose() * E_gt * p1 << std::endl;
    }

    return 0;
}