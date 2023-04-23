//
// Created by sparkai on 23-4-20.
//

#include "utils.h"

void extract_sift_feature(Mat &src, std::vector<KeyPoint> &keypoints, Mat &descriptors, int nfeature) {
    Ptr<SIFT> siftPtr = SIFT::create(nfeature);
    siftPtr->detectAndCompute(src, noArray(), keypoints, descriptors);
}

void filterMatchedPoint(Mat &descriptors1, Mat &descriptors2, std::vector<DMatch> &out, float thresh) {
    std::vector<std::vector<DMatch>> matches;
    Ptr<BFMatcher> matcherPtr = BFMatcher::create(NORM_L2);
    matcherPtr->knnMatch(descriptors1, descriptors2, matches, 2);
    for (const auto &m: matches) {
        if (m[0].distance < thresh * m[1].distance) {
            out.push_back(m[0]);
        }
    }
//    sort(out.begin(), out.end(), [](DMatch a, DMatch b) { return a.distance < b.distance; });
}


void translate_normal_camera_point(const std::vector<Point2f> &points,
                                   std::vector<Point2f> &norm_points, const Mat &K) {

    Mat inv_k = K.inv();

    const float k00 = inv_k.at<float>(0, 0);
    const float k01 = inv_k.at<float>(0, 1);
    const float k02 = inv_k.at<float>(0, 2);
    const float k10 = inv_k.at<float>(1, 0);
    const float k11 = inv_k.at<float>(1, 1);
    const float k12 = inv_k.at<float>(1, 2);
    const float k20 = inv_k.at<float>(2, 0);
    const float k21 = inv_k.at<float>(2, 1);
    const float k22 = inv_k.at<float>(2, 2);

    for (auto &p: points) {
        const float px = k00 * p.x + k01 * p.y + k02;
        const float py = k10 * p.x + k11 * p.y + k12;
        const float pz = k20 * p.x + k21 * p.y + k22;

        const Point2f norm_p{px / pz, py / pz};

        norm_points.push_back(norm_p);
    }
}

void center_and_normalize_point(const std::vector<Vector2d> &points, std::vector<Vector2d> &norm_points,
                                Matrix3d &norm_matrix) {
    Vector2d centroid{0, 0};
    double mean_dist = 0;

    for (const Vector2d &point: points) {
        centroid += point;
    }
    centroid /= points.size();

    for (const Vector2d &point: points) {
        mean_dist += (point - centroid).squaredNorm();
    }
    mean_dist = sqrt(mean_dist / points.size());
    const double factor = sqrt(2.0) / mean_dist;

    for (const Vector2d &point: points) {
        Vector2d xy;
        xy << (point - centroid) * factor;
//        std::cout << "center and normalized point dist: " << sqrt((xy - centroid).squaredNorm()) << "\n";
        norm_points.push_back(xy);
    }
    norm_matrix << factor, 0, -factor * centroid(0),
            0, factor, -factor * centroid(1),
            0, 0, 1;
}

Matrix3d estimate_essential_matrix(const std::vector<Vector2d> &points1,
                                   const std::vector<Vector2d> &points2) {

    Matrix3d E, norm_matrix1, norm_matrix2;
    std::vector<Vector2d> norm_points1, norm_points2;
    MatrixXd W(points1.size(), 9), F1, F2;

    center_and_normalize_point(points1, norm_points1, norm_matrix1);
    center_and_normalize_point(points2, norm_points2, norm_matrix2);

    for (int i = 0; i < norm_points1.size(); ++i) {
        const double u = norm_points1[i](0);
        const double v = norm_points1[i](1);
        const double up = norm_points2[i](0);
        const double vp = norm_points2[i](1);
        W.row(i) << u * up, v * up, up, u * vp, v * vp, vp, u, v, 1;
    }
    JacobiSVD<MatrixXd> svd;
    svd.compute(W, Eigen::ComputeThinV | Eigen::ComputeThinU);
    F1 = svd.matrixV().col(svd.matrixV().cols() - 1);
    F1.resize(3, 3);
    F1.transposeInPlace();

    svd.compute(F1, Eigen::ComputeThinV | Eigen::ComputeThinU);
    auto S = svd.singularValues();
    S[2] = 0;
    F2 = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

    E = norm_matrix2.transpose() * F2 * norm_matrix1;

    E.normalize();
    return E;
}


std::vector<uint> sample_some_int(int first, int last, int k, RNG rng) {
    std::vector<uint> out;
    for (int i = 0; i < k; ++i) {
        label:
        const uint random_num = rng() % last;
        for (const auto x: out) {
            if (random_num == x) goto label;
        }
        out.push_back(random_num);
    }
    return out;
}

int calculate_inliers(const Matrix3d &E, std::vector<Point2f> &pts1, std::vector<Point2f> &pts2, double t,
                      std::vector<uchar> &inliers) {
    int num_inliers = 0;
    // l' = E * p
    for (int i = 0; i < pts1.size(); ++i) {
        const Vector3d p1{pts1[i].x, pts1[i].y, 1};
        const Vector3d p2{pts2[i].x, pts2[i].y, 1};
        const Vector3d L = E * p1;
        // calculate distance from a point to a line
        const double d = abs(L.dot(p2)) / sqrt(L.squaredNorm());
        if (d < t) {
            ++num_inliers;
            inliers.push_back(true);
        } else {
            inliers.push_back(false);
        }

    }
    return num_inliers;
}


void image_show(Mat &src, std::vector<KeyPoint> &keypoints) {
    Mat img_keypoints;
    drawKeypoints(src, keypoints, img_keypoints);
    imshow("Key points Image", img_keypoints);
    waitKey();
}


void image_show(Mat &img1, std::vector<KeyPoint> &keypoints1, Mat &img2, std::vector<KeyPoint> &keypoints2,
                std::vector<DMatch> &matches, int numMatches) {
    Mat image;
    std::sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) { return a.distance < b.distance; });
    std::vector<DMatch> show_match{matches.begin(), matches.begin() + numMatches};
    drawMatches(img1, keypoints1, img2, keypoints2, show_match, image);
    imshow("Matches Image", image);
    waitKey();
}


void image_show(Mat &img1, std::vector<Point2f> &pts1, Mat &img2, std::vector<Point2f> &pts2,
                std::vector<DMatch> &matches, int numMatches) {
    Mat image;
    std::vector<KeyPoint> keypoints1, keypoints2;
    std::sort(matches.begin(), matches.end(), [](DMatch a, DMatch b) { return a.distance < b.distance; });
    std::vector<DMatch> show_match{matches.begin(), matches.begin() + numMatches};

    cv::KeyPoint::convert(pts1, keypoints1);
    KeyPoint::convert(pts2, keypoints2);

    drawMatches(img1, keypoints1, img2, keypoints2, show_match, image);
    imshow("Matches Image", image);
    waitKey();

}

