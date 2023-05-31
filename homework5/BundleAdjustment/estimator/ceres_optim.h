#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "EigenQuaternionParameterization.h"

#include "common_include.h"
#include "frame.h"
#include "map.h"
#include "feature.h"

namespace sfm {

    class ReprojectionErrorAutoDiff {
    public:

        ReprojectionErrorAutoDiff(const Eigen::Vector2d &observed_p,
                                  double fx_, double fy_,
                                  double cx_, double cy_)
                : m_observed_p(observed_p),
                  fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

        // variables: camera extrinsics and point position
        template<typename T>
        bool operator()(const T *const q,
                        const T *const t,
                        const T *const P_w,
                        T *residuals) const {
            /////////////////////// homework2 ////////////////////////////
            Eigen::Matrix<T, 2, 1> P = m_observed_p.cast<T>();
            T observed_p[2];
            observed_p[0] = T(P(0));
            observed_p[1] = T(P(1));

            // Convert quaternion from Eigen convention (x, y, z, w)
            // to Ceres convention (w, x, y, z)
            T q_ceres[4] = {q[3], q[0], q[1], q[2]};
            T Pc[3];
            ceres::QuaternionRotatePoint(q_ceres, P_w, Pc);
            Pc[0] += t[0];
            Pc[1] += t[1];
            Pc[2] += t[2];

            T u = Pc[0] / Pc[2];
            T v = Pc[1] / Pc[2];

            T predicted_x = fx * u + cx;
            T predicted_y = fy * v + cy;

            residuals[0] = predicted_x - observed_p[0];
            residuals[1] = predicted_y - observed_p[1];

            /////////////////////// homework2 ////////////////////////////
            return true;
        }


        // observed 2D point
        Eigen::Vector2d m_observed_p;
        // Camera intrinsic
        double fx, fy, cx, cy;
    };

    class TwoFrameBundleAdjuster {
    public:

        TwoFrameBundleAdjuster(double fx_, double fy_,
                               double cx_, double cy_)
                : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

        void SetLastFrame(Frame::Ptr last_frame) {
            last_frame_ = last_frame;
        }

        void SetCurrentFrame(Frame::Ptr current_frame) {
            current_frame_ = current_frame;
        }

        void SetMap(Map::Ptr map) { map_ = map; }

        void Optimize();

    private:

        Frame::Ptr last_frame_ = nullptr;
        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
        std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;

        double fx, fy, cx, cy;
    };


    class BundleAdjuster {
    public:

        BundleAdjuster(double fx_, double fy_,
                       double cx_, double cy_)
                : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

        void SetMap(Map::Ptr map) { map_ = map; }

        void Optimize();

    private:

        Frame::Ptr last_frame_ = nullptr;
        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
        std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;

        double fx, fy, cx, cy;
    };


}