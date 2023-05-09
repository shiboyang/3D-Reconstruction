#include "camera.h"
#include "visualizer.h"
#include "essential_matrix.h"
#include "stereo_rectify.h"
#include "epipolar_search.h"

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

    Eigen::Matrix3d H1, H2;

    Eigen::Matrix4d Tc2c1 = Twc2.inverse() * Twc1;

    Eigen::Quaterniond Qc2c1 = Eigen::Quaterniond(Tc2c1.block<3, 3>(0, 0));
    Eigen::Vector3d tc2c1 = Tc2c1.block<3, 1>(0, 3);

    RectifyStereoCamerasByPoints(camera, normal_points1, normal_points2, points1, points2, &H1, &H2);

    std::cout << "H1" << std::endl;
    std::cout << H1 << std::endl;

    std::cout << "H2" << std::endl;
    std::cout << H2 << std::endl;

    cv::Mat out1(camera.image_h, camera.image_w, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat out2(camera.image_h, camera.image_w, CV_8UC3, cv::Scalar(255, 255, 255));
    // homework2: 得到用于双目立体矫正的H矩阵后，将对应的图像矫正到平行视图。
    cv::Mat img1 = camera.GetFirstFrameImage();
    cv::Mat img2 = camera.GetSecondFrameImage();

    cv::Mat M1, M2;
    cv::eigen2cv(H1, M1);
    cv::eigen2cv(H2, M2);
    cv::warpPerspective(img1, out1, M1, out1.size());
    cv::warpPerspective(img2, out2, M2, out2.size());

    apply_holograph_matrix(points1, H1);
    apply_holograph_matrix(points2, H2);

    /////// end homework2 end ///////

    cv::imshow("warp1", out1);
    cv::waitKey(0);

    cv::imshow("warp2", out2);
    cv::waitKey(0);


    std::vector<cv::Point2f> points1_cv = convertPointsOfEigenToCV(points1);
    std::vector<cv::Point2f> points2_cv = convertPointsOfEigenToCV(points2);

    std::vector<cv::Point2f> points1_trans;
    std::vector<cv::Point2f> points2_trans;

//    cv::perspectiveTransform(points1_cv, points1_trans, homo1_);
//    cv::perspectiveTransform(points2_cv, points2_trans, homo2_);

    std::vector<Eigen::Vector3i> disparity;
    disparity = DisparityPointsCalculate(points1_trans, points2_trans);

    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    K(0, 0) = camera.fx;
    K(1, 1) = K(0, 0);
    K(0, 2) = camera.cx;
    K(1, 2) = camera.cy;

    double baseline = tc2c1.norm();

    std::vector<Eigen::Vector3d> est_points = Estimate3DPoints(disparity, Twc1, K, H1, baseline);

    int frame_width = 1920;
    int frame_height = 1080;

    pangolin::CreateWindowAndBind("PoseViewer", frame_width, frame_height);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(frame_width, frame_height, (frame_width + frame_height) / 2,
                                       (frame_width + frame_height) / 2, frame_width / 2, frame_height / 2,
                                       0.0001, 100000),
            pangolin::ModelViewLookAt(-10, -10, 0, 0, 0, 0, pangolin::AxisZ)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -float(frame_width) / frame_height)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        DrawModel(camera.ModelPoints, camera.ModelLines);

        DrawRawFrame(Twc1);
        DrawRawFrame(Twc2);

        DrawEstimatedModel(est_points);

        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }


    return 0;
}