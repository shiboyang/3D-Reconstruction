#pragma once

#include "common_include.h"

namespace sfm {

struct Frame;

struct Feature;

/**
 * 路标点类
 * 特征点在三角化之后形成路标点
 */
struct MapPoint {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id_ = 0;  // ID
    bool is_outlier_ = false;
    bool optimized = false;

    Vec3 pos_ = Vec3::Zero();  // Position in world
    Vec3 init_pos_ = Vec3::Zero();  // Position without BA
    Vec3 rgb_ = Vec3::Zero();  // Point's RGB
    std::mutex data_mutex_;
    int observed_times_ = 0;  // being observed by feature matching algo.
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint() {}

    MapPoint(long id, Vec3 position);

    MapPoint(long id, Vec3 position, Vec3 rgb);

    Vec3 RGB() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return rgb_;
    }

    void SetRGB(const Vec3 &rgb) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        rgb_ = rgb;
    };

    Vec3 Pos() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void SetPos(const Vec3 &pos) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    };

    void SetBAPos(const Vec3 &pos) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        init_pos_ = pos_;
        pos_ = pos;
    };

    void AddObservation(std::shared_ptr<Feature> feature) {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feat);

    std::list<std::weak_ptr<Feature>> GetObs() {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    // factory function
    static MapPoint::Ptr CreateNewMappoint();
};
}  // namespace sfm

