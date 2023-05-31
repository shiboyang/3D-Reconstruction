#include "ceres_optim.h"

namespace sfm {

    void TwoFrameBundleAdjuster::Optimize() {
        ceres::Problem problem;

        // 原始的Pose应该是 Twc，从相机到世界坐标系的定义
        Eigen::Quaterniond q_c1_w = Eigen::Quaterniond(last_frame_->Pose().inverse().linear());
        Eigen::Vector3d t_c1_w = Eigen::Vector3d(0.001, 0.001, 0.001);

        Eigen::Quaterniond q_c2_w = Eigen::Quaterniond(current_frame_->Pose().inverse().linear());
        Eigen::Vector3d t_c2_w = current_frame_->Pose().inverse().translation();

        std::vector<Eigen::Vector3d> points;
        points.clear();
        std::vector<int> points_idx;
        points_idx.clear();

        active_landmarks_ = map_->GetActiveMapPoints();

        for (auto &landmark: active_landmarks_) {
            points_idx.push_back(landmark.first);
            auto pos = landmark.second->Pos();
            points.push_back(pos);
        }

        for (int i = 0; i < points.size(); i++) {

            for (auto &feature: active_landmarks_[points_idx[i]]->observations_) {

                auto feat = feature.lock();

                Eigen::Vector2d observed_p(feat->position_.pt.x, feat->position_.pt.y);

                auto frame = feat->frame_.lock();

                if (frame->id_ == 0) {
                    ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 4, 3, 3>(
                            new ReprojectionErrorAutoDiff(observed_p, fx, fy, cx, cy));

                    ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             q_c1_w.coeffs().data(), t_c1_w.data(), points.at(i).data());
                } else {
                    ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 4, 3, 3>(
                            new ReprojectionErrorAutoDiff(observed_p, fx, fy, cx, cy));

                    ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);

                    problem.AddResidualBlock(costFunction, lossFunction,
                                             q_c2_w.coeffs().data(), t_c2_w.data(), points.at(i).data());
                }
            }

        }

        ceres::LocalParameterization *quaternionParameterization =
                new EigenQuaternionParameterization;

        problem.SetParameterization(q_c1_w.coeffs().data(), quaternionParameterization);
        problem.SetParameterization(q_c2_w.coeffs().data(), quaternionParameterization);

        // 把第一帧的位姿设置为Identity
        problem.SetParameterBlockConstant(q_c1_w.coeffs().data());
        problem.SetParameterBlockConstant(t_c1_w.data());

        ceres::Solver::Options options;
        options.max_num_iterations = 1000;
        options.num_threads = 4;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.BriefReport() << std::endl;

        for (int i = 0; i < points_idx.size(); i++) {
            active_landmarks_[points_idx[i]]->SetBAPos(points[i]);
            active_landmarks_[points_idx[i]]->optimized = true;
        }

        Eigen::Affine3d Tc2w_update;
        Tc2w_update.linear() = q_c2_w.toRotationMatrix();
        Tc2w_update.translation() = t_c2_w;

        current_frame_->SetBAPose(Tc2w_update.inverse());
        current_frame_->optimized = true;

        std::cout << "end optim " << std::endl;
    }

    void BundleAdjuster::Optimize() {
        ceres::Problem problem;
        ceres::LocalParameterization *quaternionParameterization =
                new EigenQuaternionParameterization;

        active_keyframes_ = map_->GetActiveKeyFrames();
        active_landmarks_ = map_->GetActiveMapPoints();

        ////////////////////////// homework3 ////////////////////////

        // 原始的Pose应该是 Twc，从相机到世界坐标系的定义
        // 第一帧Pose
        Eigen::Quaterniond q_c1_w = Eigen::Quaterniond(active_keyframes_.at(0)->Pose().inverse().linear());
        Eigen::Vector3d t_c1_w = Eigen::Vector3d(0.001, 0.001, 0.001);

        std::unordered_map<unsigned long, Eigen::Quaterniond> quaternions_map;
        std::unordered_map<unsigned long, Eigen::Vector3d> translates_map;
        for (auto &frame: active_keyframes_) {
            if (frame.first == 0) continue;
            // Twc -> Tcw
            Eigen::Quaterniond q_c_w = Eigen::Quaterniond(frame.second->Pose().inverse().linear());
            Eigen::Vector3d t_c_w = frame.second->Pose().inverse().translation();
            quaternions_map.insert(std::make_pair(frame.first, q_c_w));
            translates_map.insert(std::make_pair(frame.first, t_c_w));
        }

        std::vector<Eigen::Vector3d> points;
        points.clear();
        std::vector<uint> points_idx;
        points_idx.clear();

        for (const auto &landmark: active_landmarks_) {
            points_idx.push_back(landmark.first);
            points.push_back(landmark.second->Pos());
        }

        for (int i = 0; i < points.size(); ++i) {
            for (auto &feature: active_landmarks_[points_idx[i]]->GetObs()) {
                auto feat = feature.lock();
                Eigen::Vector2d observed_p(feat->position_.pt.x, feat->position_.pt.y);
                auto frame = feat->frame_.lock();

                if (frame->id_ == 0) {
                    ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 4, 3, 3>(
                            new ReprojectionErrorAutoDiff(observed_p, fx, fy, cx, cy));
                    ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
                    problem.AddResidualBlock(costFunction, lossFunction,
                                             q_c1_w.coeffs().data(), t_c1_w.data(), points.at(i).data());
                } else {
                    ceres::CostFunction *costFunction = new ceres::AutoDiffCostFunction<ReprojectionErrorAutoDiff, 2, 4, 3, 3>(
                            new ReprojectionErrorAutoDiff(observed_p, fx, fy, cx, cy));
                    ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
                    problem.AddResidualBlock(costFunction, lossFunction,
                                             quaternions_map.at(frame->id_).coeffs().data(),
                                             translates_map.at(frame->id_).data(),
                                             points.at(i).data());
                }
            }
        }

        for (const auto &frame: active_keyframes_) {
            unsigned long frame_id = frame.first;
            if (frame_id == 0) {
                problem.SetParameterization(q_c1_w.coeffs().data(), quaternionParameterization);
            } else {
                problem.SetParameterization(quaternions_map.at(frame_id).coeffs().data(), quaternionParameterization);
            }
        }

        // 把第一帧的位姿设置为Identity
        problem.SetParameterBlockConstant(q_c1_w.coeffs().data());
        problem.SetParameterBlockConstant(t_c1_w.data());

        ceres::Solver::Options options;
        options.max_num_iterations = 1000;
        options.num_threads = 4;
        options.minimizer_progress_to_stdout = true;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << summary.BriefReport() << std::endl;

        for (int i = 0; i < points_idx.size(); i++) {
            active_landmarks_[points_idx[i]]->SetBAPos(points[i]);
            active_landmarks_[points_idx[i]]->optimized = true;
        }

        for (const auto &frame: active_keyframes_) {
            unsigned long frame_id = frame.first;
            if (frame_id == 0) continue;
            Eigen::Affine3d Tcw_update;
            Tcw_update.linear() = quaternions_map.at(frame_id).toRotationMatrix();
            Tcw_update.translation() = translates_map.at(frame_id);

            frame.second->SetBAPose(Tcw_update.inverse());
            frame.second->optimized = true;
        }

        //////////////////////////// homework3 //////////////////////////
    }

}
