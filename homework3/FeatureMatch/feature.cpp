#include "utils.h"


using namespace std;
using namespace cv;

// RANSAC参数
const int RANSAC_ITERATIONS = 1000;
const double RANSAC_THRESHOLD = 1.0;

// 伪随机数生成器
RNG rng(12345);

// RANSAC求解Essential Matrix
Mat findEssentialMatrixRANSAC(const vector<Point2f> &pts1, const vector<Point2f> &pts2, const Mat &K,
                              vector<uchar> &inliers) {
    int N = pts1.size();
    int max_inliers = 0;
    Mat best_E;
    const float p = 0.99;
    vector<Vector2d> sample_points1, sample_points2;
    Matrix3d essential_matrix, best_e;
    vector<Point2f> norm_pts1, norm_pts2;

    // from pixel coordinate to camera coordinate
    // for normal camera:
    //   (Pc_x, Pc_y, 1)(camera coordinate) == (Pw_x, Pw_y, Pw_z)world coordinate
    translate_normal_camera_point(pts1, norm_pts1, K);
    translate_normal_camera_point(pts2, norm_pts2, K);

    for (int iter = 0; iter < RANSAC_ITERATIONS; ++iter) {
        // homework2: 选择8个匹配点，并完成Essential matrix的计算
        sample_points1.clear();
        sample_points2.clear();

        for (const auto &i: sample_some_int(0, N, 8, rng)) {
            const Vector2d p1{norm_pts1[i].x, norm_pts1[i].y};
            const Vector2d p2{norm_pts2[i].x, norm_pts2[i].y};
            sample_points1.push_back(p1);
            sample_points2.push_back(p2);
        }
        essential_matrix = estimate_essential_matrix(sample_points1, sample_points2);
        // homework2 end

        // homework3: 计算内点数量
        int num_inliers = calculate_inliers(essential_matrix, norm_pts1, norm_pts2, RANSAC_THRESHOLD,
                                            inliers);
        // homework3 end

        // homework4: 如果当前内点数量大于最大内点数量，则更新最佳Essential Matrix
        if (num_inliers > max_inliers) {
            max_inliers = num_inliers;
            const double e = 1.0 - num_inliers / norm_pts1.size();
            N = log(1 - p) / log(1 - pow(1 - e, 8));
            best_e = essential_matrix;
            if (iter >= N) break;

        }
        // homework4 end
    }

    sample_points1.clear();
    sample_points2.clear();

    for (int i = 0; i < norm_pts1.size(); ++i) {
        if (inliers[i]) {
            const Vector2d p1{norm_pts1[i].x, norm_pts1[i].y};
            const Vector2d p2{norm_pts2[i].x, norm_pts2[i].y};
            sample_points1.push_back(p1);
            sample_points2.push_back(p2);
        }
    }
    essential_matrix = estimate_essential_matrix(sample_points1, sample_points2);
    cv::eigen2cv(essential_matrix, best_E);

    return best_E;
}

int main() {
    // 加载图像并提取特征点
    Mat img1 = imread("../1.JPG", IMREAD_GRAYSCALE);
    Mat img2 = imread("../2.JPG", IMREAD_GRAYSCALE);
//    Mat img1 = imread("../n1.JPG", IMREAD_GRAYSCALE);
//    Mat img2 = imread("../n2.JPG", IMREAD_GRAYSCALE);


    if (img1.empty() || img2.empty()) {
        cerr << "Error loading images!" << endl;
        return -1;
    }

    vector<Point2f> pts1, pts2;

    // homework1: 提取SIFT特征，并进行特征匹配，可以调用OpenCV的函数
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    std::vector<DMatch> matches;

    extract_sift_feature(img1, keypoints1, descriptors1);
    extract_sift_feature(img2, keypoints2, descriptors2);
    std::cout << "img1 descriptors shape: " << descriptors1.rows << "x" << descriptors1.cols << "\n";
    std::cout << "img2 descriptors shape: " << descriptors2.rows << "x" << descriptors2.cols << "\n";
    filterMatchedPoint(descriptors1, descriptors2, matches, 0.6);
    std::cout << "matched point :" << matches.size() << "\n";

    for (const DMatch &match: matches) {
        pts1.push_back(keypoints1[match.queryIdx].pt);
        pts2.push_back(keypoints2[match.trainIdx].pt);
    }
    // image_show(img1, keypoints1, img2, keypoints2, matches, 10);
    // homework1 end


    // 假设相机内参矩阵已知
    Mat K = (Mat_<float>(3, 3) << 2556, 0, 1536, 0, 2556, 1152, 0, 0, 1);

    // 使用RANSAC估计Essential Matrix
    vector<uchar> inliers;
    Mat E = findEssentialMatrixRANSAC(pts1, pts2, K, inliers);

    cout << "Essential Matrix:" << endl << E << endl;

    Mat E_cv = findEssentialMat(pts1, pts2, K, RANSAC, 0.99, 1.0);

    cout << "Essential Matrix From OpenCV:" << endl << E_cv << endl;

    // 可视化内点匹配
    vector<DMatch> inlier_matches;

    // 由于特征点匹配过于密集，可视化的效果不够明显，可以将 i++ 改为 i += 20
    for (size_t i = 0; i < inliers.size(); i+=200) {
        if (inliers[i]) {
            inlier_matches.push_back(matches[i]);
        }
    }

    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, img_matches);
    cv::namedWindow("Inlier Matches", cv::WINDOW_NORMAL);
    imshow("Inlier Matches", img_matches);
    resizeWindow("Inlier Matches", 1080, 960);
    waitKey();

    return 0;
}
