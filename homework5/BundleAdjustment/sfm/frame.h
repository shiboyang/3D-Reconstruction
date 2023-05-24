#pragma once

#include "camera.h"
#include "common_include.h"

namespace sfm {

// forward declare
struct MapPoint;
struct Feature;

/**
 * 帧
 * 每一帧分配独立id，关键帧分配关键帧ID
 */
struct Frame {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;           // id of this frame
    unsigned long keyframe_id_ = 0;  // id of key frame
    bool is_keyframe_ = false;       // 是否为关键帧
    double time_stamp_;              // 时间戳，暂不使用
    SE3 pose_;                       // Twc 形式Pose
    SE3 init_pose_;                  // 没有经过BA优化之前的pose
    std::mutex pose_mutex_;          // Pose数据锁
    cv::Mat img_;                    // image
    cv::Mat rgb_img_;                // rgb image

    // extracted features in image
    std::vector<std::shared_ptr<Feature>> features_;
    cv::Mat descriptors_;

    bool optimized = false;

   public:  // data members
    Frame() {}

    Frame(long id, double time_stamp, const SE3 &pose, const Mat &img);

    // set and get pose, thread safe
    SE3 Pose() {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    void SetBAPose(const SE3 &pose) {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        init_pose_ = pose_;
        pose_ = pose;
    }

    /// 设置关键帧并分配并键帧id
    void SetKeyFrame();

    /// 工厂构建模式，分配id 
    static std::shared_ptr<Frame> CreateFrame();
};

}  // namespace