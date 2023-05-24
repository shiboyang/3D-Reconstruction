#include "dataset.h"
#include "frame.h"

#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>
using namespace std;

namespace sfm {

Dataset::Dataset(const std::string& dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::Init() {
    // read camera intrinsics and extrinsics
    ifstream f(dataset_path_ + "/calib.txt");
    double fx, fy, cx, cy, rad;
    std::string s;
    std::getline(f,s);
    if(!s.empty())
    {
        std::stringstream ss;
        ss << s;
        
        ss >> fx;
        ss >> cx;
        ss >> cy;
        ss >> rad;

        fy = fx;
    }
    Mat33 K;
    K << fx, 0, cx,
        0, fy, cy,
        0, 0, 1;

    K = K * 0.5;

    Eigen::Affine3d init;
    init.setIdentity();

    Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2), init));
    camera_ = new_camera;

    f.close();

    current_image_index_ = 1;
    return true;
}

Frame::Ptr Dataset::NextFrame() {
    boost::format fmt("%s/%d.JPG");
    cv::Mat image_;
    // read images
    image_ = cv::imread((fmt % dataset_path_ % current_image_index_).str(), cv::IMREAD_GRAYSCALE);

    cv::Mat rgb_image_;
    rgb_image_ = cv::imread((fmt % dataset_path_ % current_image_index_).str(), cv::IMREAD_COLOR);

    cv::Mat image_resized;
    cv::resize(image_, image_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

    cv::Mat rgb_image_resized;
    cv::resize(rgb_image_, rgb_image_resized, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);

    auto new_frame = Frame::CreateFrame();
    new_frame->img_ = image_resized;
    new_frame->rgb_img_ = rgb_image_resized;
    
    current_image_index_++;
    return new_frame;
}

}  // namespace myslam