#include <opencv2/opencv.hpp>

#include "backend.h"
#include "config.h"
#include "feature.h"
#include "frontend.h"
#include "map.h"
#include "viewer.h"

#include "camera.h"
#include "utils.h"
#include "types.h"
#include "essential_matrix.h"
#include "ceres_optim.h"

namespace sfm {

Frontend::Frontend() {
    sift_detector = cv::SIFT::create();
    num_features_init_ = Config::Get<int>("num_features_init");
    num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(sfm::Frame::Ptr frame) {
    std::cout << "Add Frame"<<std::endl;
    current_frame_ = frame;
    int num_features = DetectFeatures();

    if(last_frame_ == nullptr) {
        std::cout << "Last Frame is NULL" << std::endl;
        last_frame_ = current_frame_;
        return true;
    }

    switch (status_) {
        case FrontendStatus::INITING:
            std::cout << "init "<<std::endl;
            Init();
            break;
        case FrontendStatus::TRACKING_GOOD:
            std::cout << "track good  "<<std::endl;
            Track();
            break;
        case FrontendStatus::TRACKING_BAD:
            std::cout << "track bad  "<<std::endl;
            Track();
            break;
        case FrontendStatus::LOST:
            std::cout << "lost "<<std::endl;
            Reset();
            break;

        std::cout << "unknow "<<std::endl;
    }

    last_frame_ = current_frame_;
    return true;
}

////////////////////////////////////////////初始化部分///////////////////////////////

bool Frontend::Init() {
    if(current_frame_ == nullptr || last_frame_ == nullptr) {
        std::cout << "need more than 2 frames"<<std::endl;
        return false;
    }

    // 初始的位姿都设置为 0
    Eigen::Affine3d init_pose;
    init_pose.setIdentity();

    last_frame_->SetPose(init_pose);
    current_frame_->SetPose(init_pose);

    std::vector<cv::Point2f> cv_points1, cv_points2;
    std::vector<Eigen::Vector2d> points1;
    std::vector<Eigen::Vector2d> points2;

    MatchFeatures(last_frame_->descriptors_, current_frame_->descriptors_, matches);

    for (const auto& match : matches) {
        cv_points1.push_back(last_frame_->features_[match.queryIdx]->position_.pt);
        cv_points2.push_back(current_frame_->features_[match.trainIdx]->position_.pt);
    }

    cv::Mat cv_R, cv_t, mask;
    cv::Mat cv_intrinsic_;
    cv::eigen2cv(camera_->K(), cv_intrinsic_);
    // 为了避免Essential矩阵求解实现过程和RANSAC实现过程存在问题，这个地方直接调用opencv的结果
    cv::Mat cv_Essential = cv::findEssentialMat(cv_points1, cv_points2, cv_intrinsic_, cv::RANSAC, 0.999, 1.0, mask);
    
    // 保存匹配成功并且通过ransac筛选之后的点
    for(int match_idx = 0; match_idx < matches.size(); match_idx ++) {
        // 根据mask的值判断在findEssential中这个点是不是外点
        if(int(mask.at<u_char>(match_idx)) == 0) continue;

        auto match = matches[match_idx];
        points1.push_back(Eigen::Vector2d(last_frame_->features_[match.queryIdx]->position_.pt.x,
                                          last_frame_->features_[match.queryIdx]->position_.pt.y) );
        points2.push_back(Eigen::Vector2d(current_frame_->features_[match.trainIdx]->position_.pt.x, 
                                          current_frame_->features_[match.trainIdx]->position_.pt.y) );

        last_frame_->features_[match.queryIdx]->inlier_ = true;
        current_frame_->features_[match.trainIdx]->inlier_ = true;

        inlier_matches.push_back(std::make_pair(match.queryIdx, match.trainIdx));
    }

    std::cout << "match size is "<<matches.size()<<std::endl 
              << "inlier size is "<< inlier_matches.size()<<std::endl;
    
    // 如果没有实现PoseFromEssentialMatrix，可以调用opencv的recoverPose函数求解R、t
    // cv::recoverPose(cv_Essential, cv_points1, cv_points2, cv_intrinsic_, cv_R, cv_t, mask);

    Eigen::Matrix3d eigen_Essential;
    cv::cv2eigen(cv_Essential, eigen_Essential);

    // R_c2_c1 表达的是从camera1 to camera2的旋转
    Eigen::Matrix3d R_c2_c1;
    Eigen::Vector3d t_c2_c1;
    std::vector<Eigen::Vector3d> points3d_cv;

    PoseFromEssentialMatrix(eigen_Essential, camera_->K(), points1, points2, &R_c2_c1, &t_c2_c1, &points3d_cv);

    // cv::cv2eigen(cv_R, R_c2_c1);
    // cv::cv2eigen(cv_t, t_c2_c1);

    Eigen::Affine3d T_update;
    T_update.linear() = R_c2_c1;
    T_update.translation() = t_c2_c1;
    T_update = T_update.inverse();
    Eigen::Affine3d T_c2 = current_frame_->Pose();
    T_c2 = T_c2 * T_update;
    current_frame_->SetPose(T_c2);

    bool build_map_success = BuildInitMap();
    OptimizeCurrentPose();
    if (build_map_success) {
        status_ = FrontendStatus::TRACKING_GOOD;
        if (viewer_) {
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
        return true;
    }
    return false;
}

bool Frontend::BuildInitMap() {
    std::cout << "Start BuildInitMap"<<std::endl;

    Eigen::Matrix4d Twc1, Twc2, Tcw1, Tcw2;

    Twc1 = last_frame_->Pose().matrix();
    Twc2 = current_frame_->Pose().matrix();

    Tcw1 = Twc1.inverse();
    Tcw2 = Twc2.inverse();

    Eigen::Matrix3x4d proj_mat1 = GetProjectMatrix(camera_->K(), Tcw1);
    Eigen::Matrix3x4d proj_mat2 = GetProjectMatrix(camera_->K(), Tcw2);

    std::vector<Eigen::Vector2d> inliers1;
    std::vector<Eigen::Vector2d> inliers2;

    int n_points = inlier_matches.size();

    for(int i = 0; i < n_points; i++)
    {
        auto match = inlier_matches[i];
        inliers1.push_back(Eigen::Vector2d(last_frame_->features_[match.first]->position_.pt.x,
                                           last_frame_->features_[match.first]->position_.pt.y) );
        inliers2.push_back(Eigen::Vector2d(current_frame_->features_[match.second]->position_.pt.x, 
                                           current_frame_->features_[match.second]->position_.pt.y) );
    }
        
    std::vector<Eigen::Vector3d> triangulation_results = TriangulatePoints(proj_mat1, proj_mat2, inliers1, inliers2);

    for(int i = 0; i < n_points; i++) {
        Eigen::Vector3d point_3d = triangulation_results[i];
        if(point_3d(2) < 0.1) {
            last_frame_->features_[inlier_matches[i].first]->inlier_ = false;
            current_frame_->features_[inlier_matches[i].second]->inlier_ = false;
            continue;
        }

        int row = int(last_frame_->features_[inlier_matches[i].first]->position_.pt.y);
        int col = int(last_frame_->features_[inlier_matches[i].first]->position_.pt.x);

        Eigen::Vector3d color;
        color.x() = last_frame_->rgb_img_.at<cv::Vec3b>(row, col)[0];
        color.y() = last_frame_->rgb_img_.at<cv::Vec3b>(row, col)[1];
        color.z() = last_frame_->rgb_img_.at<cv::Vec3b>(row, col)[2];

        auto new_map_point = MapPoint::CreateNewMappoint();
        new_map_point->SetPos(point_3d);
        new_map_point->SetRGB(color);
        new_map_point->AddObservation(last_frame_->features_[inlier_matches[i].first]);
        new_map_point->AddObservation(current_frame_->features_[inlier_matches[i].second]);

        last_frame_->features_[inlier_matches[i].first]->map_point_ = new_map_point;
        current_frame_->features_[inlier_matches[i].second]->map_point_ = new_map_point;
        
        map_->InsertMapPoint(new_map_point);
    }

    
    last_frame_->SetKeyFrame();
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(last_frame_);
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    return true;
}

int Frontend::OptimizeCurrentPose() {
    TwoFrameBundleAdjuster ba = TwoFrameBundleAdjuster(camera_->fx_, camera_->fy_, 
                                                       camera_->cx_, camera_->cy_);
    ba.SetLastFrame(last_frame_);
    ba.SetCurrentFrame(current_frame_);
    ba.SetMap(map_);

    ba.Optimize();
    return 0;
}

////////////////////////////////// 初始化部分 ////////////////////////////////


////////////////////////////////// Tracking 部分 ////////////////////////////////

bool Frontend::Track() {
    if (last_frame_) {
        current_frame_->SetPose(last_frame_->Pose());
    }

    std::cout << "Start to Track Last Frame"<<std::endl;

    int num_track_last = TrackLastFrame();
    tracking_inliers_ = num_track_last;
    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        std::cout << "Lost with "<<num_track_last<<std::endl;
        status_ = FrontendStatus::LOST;
    }

    InsertKeyframe();
    relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

    
    if (viewer_) {
        viewer_->AddCurrentFrame(current_frame_);
        viewer_->UpdateMap();
    }
    return true;
}

bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }
    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    std::cout << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_ << std::endl;

    SetObservationsForKeyFrame();
    
    // triangulate map points
    // TriangulateNewPoints();
    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;
}

void Frontend::SetObservationsForKeyFrame() {
    for (auto &feat : current_frame_->features_) {
        auto mp = feat->map_point_.lock();
        if (mp) mp->AddObservation(feat);
    }
}

int Frontend::TrackLastFrame() {
    std::vector<Eigen::Vector2d> points1, points2;
    std::vector<cv::Point2f> cv_points1, cv_points2;
    std::vector<cv::Point3f> points_3d;
    std::vector<cv::Point2f> points_2d;

    std::vector<MapPoint::Ptr> map_points;

    std::vector<int> current_3d_points_idx;

    std::vector<int> current_2d_points_last_idx;
    std::vector<int> current_2d_points_curr_idx;

    matches.clear();
    MatchFeatures(last_frame_->descriptors_, current_frame_->descriptors_, matches);

    for (const auto& match : matches) {
        auto point = last_frame_->features_[match.queryIdx]->map_point_.lock();
        if(point != nullptr) {
            // 寻找对应的3D点，直接对最新的一帧求解PnP
            Eigen::Vector3d pt = point->Pos();
            points_2d.push_back(current_frame_->features_[match.trainIdx]->position_.pt);
            points_3d.push_back(cv::Point3f(pt(0), pt(1), pt(2)));

            map_points.push_back(point);
            current_3d_points_idx.push_back(match.trainIdx);
        } else {
            // 寻找对应的2D点，三角化得到一些新的三维点
            cv_points1.push_back(last_frame_->features_[match.queryIdx]->position_.pt);
            cv_points2.push_back(current_frame_->features_[match.trainIdx]->position_.pt);

            points1.push_back(Eigen::Vector2d(last_frame_->features_[match.queryIdx]->position_.pt.x, 
                                              last_frame_->features_[match.queryIdx]->position_.pt.y));
            points2.push_back(Eigen::Vector2d(current_frame_->features_[match.trainIdx]->position_.pt.x, 
                                              current_frame_->features_[match.trainIdx]->position_.pt.y));

            current_2d_points_last_idx.push_back(match.queryIdx);
            current_2d_points_curr_idx.push_back(match.trainIdx);
        }

    }

    cv::Mat cv_r, cv_R, cv_t, mask;
    cv::Mat cv_intrinsic_;
    cv::eigen2cv(camera_->K(), cv_intrinsic_);
    cv::solvePnPRansac(points_3d, points_2d, cv_intrinsic_, cv::Mat(), cv_r, cv_t, mask);
    cv::Rodrigues(cv_r, cv_R);

    // std::cout << "mask "<<mask<<std::endl;

    Eigen::Matrix3d Rcw;
    Eigen::Vector3d tcw;
    cv::cv2eigen(cv_R, Rcw);
    cv::cv2eigen(cv_t, tcw);

    Eigen::Affine3d Tcw;
    Tcw.linear() = Rcw;
    Tcw.translation() = tcw;

    current_frame_->SetPose(Tcw.inverse());

    int num_good_pts = points_3d.size();
    std::cout << "Find " << num_good_pts << " 3D points in the last image." << std::endl;

    // 处理三维点，一个是将新的图像里边的二维点与三维点进行关联
    // 另外一个是三角化没有三维点的二维点，将这些三角化之后的点加入到map中

    for(int i = 0; i < mask.rows; i++) {
        int inlier_idx = mask.at<u_int32_t>(i);
        // 将二维点与三维点进行关联
        map_points[inlier_idx]->AddObservation(current_frame_->features_[current_3d_points_idx[inlier_idx]]);
        current_frame_->features_[current_3d_points_idx[inlier_idx]]->map_point_ = map_points[inlier_idx];
    }

    Eigen::Affine3d T_c2_c1 = current_frame_->Pose().inverse() * last_frame_->Pose();
    Eigen::Matrix3d E = EssentialMatrixFromPose(T_c2_c1.linear(), T_c2_c1.translation().normalized());

    std::vector<int> mask_match;
    mask_match.resize(points1.size());

    for(int i = 0; i < points1.size(); i++) {
        double error = points2[i].homogeneous().transpose() 
                       * camera_->K().inverse().transpose() * E * camera_->K().inverse() 
                       * points1[i].homogeneous();
        if(error < 1.0) {mask_match[i] = 1;}
        else {mask_match[i] = 0;}
    }

    Eigen::Matrix4d Twc1, Twc2, Tcw1, Tcw2;

    Twc1 = last_frame_->Pose().matrix();
    Twc2 = current_frame_->Pose().matrix();

    Tcw1 = Twc1.inverse();
    Tcw2 = Twc2.inverse();

    Eigen::Matrix3x4d proj_mat1 = GetProjectMatrix(camera_->K(), Tcw1);
    Eigen::Matrix3x4d proj_mat2 = GetProjectMatrix(camera_->K(), Tcw2);
    
    std::vector<Eigen::Vector3d> triangulation_results 
            = TriangulatePoints(proj_mat1, proj_mat2, points1, points2);

    int n_points = points1.size();
    for(int i = 0; i < n_points; i++) {
        Eigen::Vector3d point_3d = triangulation_results[i];
        if(point_3d(2) < 0.1 || mask_match[i] == 0) {
            last_frame_->features_[current_2d_points_last_idx[i]]->inlier_ = false;
            current_frame_->features_[current_2d_points_curr_idx[i]]->inlier_ = false;
            continue;
        }

        int row = int(last_frame_->features_[current_2d_points_last_idx[i]]->position_.pt.y);
        int col = int(last_frame_->features_[current_2d_points_last_idx[i]]->position_.pt.x);

        Eigen::Vector3d color;
        color.x() = last_frame_->rgb_img_.at<cv::Vec3b>(row, col)[0];
        color.y() = last_frame_->rgb_img_.at<cv::Vec3b>(row, col)[1];
        color.z() = last_frame_->rgb_img_.at<cv::Vec3b>(row, col)[2];

        auto new_map_point = MapPoint::CreateNewMappoint();
        new_map_point->SetPos(point_3d);
        new_map_point->SetRGB(color);
        new_map_point->AddObservation(last_frame_->features_[current_2d_points_last_idx[i]]);
        new_map_point->AddObservation(current_frame_->features_[current_2d_points_curr_idx[i]]);

        last_frame_->features_[current_2d_points_last_idx[i]]->map_point_ = new_map_point;
        current_frame_->features_[current_2d_points_curr_idx[i]]->map_point_ = new_map_point;
        
        map_->InsertMapPoint(new_map_point);
    }

    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);
    backend_->UpdateMap();

    BundleAdjuster ba = BundleAdjuster(camera_->fx_, camera_->fy_, 
                                       camera_->cx_, camera_->cy_);
    ba.SetMap(map_);
    ba.Optimize();

    return num_good_pts;
}


/////////////////////////////////// Tracking 部分 //////////////////////////////


//////////////////////////////////// 特征检测和匹配部分 ///////////////////////////

int Frontend::DetectFeatures() {

    std::vector<cv::KeyPoint> keypoints;
    //-- Step 1: Detect keypoints using SIFT detector
    cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
    detector->detect(current_frame_->img_, keypoints);

    //-- Step 2: Compute descriptors
    cv::Mat descriptors;
    detector->compute(current_frame_->img_, keypoints, descriptors);
    descriptors.copyTo(current_frame_->descriptors_);

    int cnt_detected = 0;
    for (auto &kp : keypoints) {
        current_frame_->features_.push_back(
            Feature::Ptr(new Feature(current_frame_, kp)));
        cnt_detected++;
    }

    std::cout << "Detect " << cnt_detected << " new features" << std::endl;
    return cnt_detected;
}

int Frontend::MatchFeatures(const cv::Mat& descriptors1,
                            const cv::Mat& descriptors2,
                            std::vector<cv::DMatch>& matches) {
    //-- Matching descriptors using FLANN matcher
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.3f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }

    return 0;
}

/////////////////////////////////// 特征检测和匹配部分 ///////////////////////////

bool Frontend::Reset() {
    return true;
}

}  // namespace sfm