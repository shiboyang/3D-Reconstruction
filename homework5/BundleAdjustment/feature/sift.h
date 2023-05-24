#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm> // for min

// keypoint class
class keypoint
{
public:
    keypoint(int oct = 0, int lyr = 0, cv::Point p = cv::Point(0, 0), double scl = 0, float ang = 0):
        octave(oct), layer(lyr), scale(scl), pt(p), angle(ang) {}

    static void coinvertToKeyPoint(std::vector<keypoint>& kpts, std::vector<cv::KeyPoint>& KPts);

public:
    int octave;
    int layer; // 注意从0开始
    cv::Point pt;
    double scale; // 尺度空间坐标
    float angle;
    float response;
    int oct_info;
};

// pyramid class
class pyramid
{
public:
    void appendTo(int oct, cv::Mat& img); // 金字塔插入图像
    void build(int oct) { pyr.resize(oct); }
    void clear() { pyr.clear(); }
    int octaves() { return pyr.size(); } // 层数 
    std::vector<cv::Mat>& operator[] (int oct); // 重载[]

private:
    std::vector<std::vector<cv::Mat> > pyr;
};

// sift class
class SIFT
{
public:
    SIFT(int s = 3, double sigma = 1.6) : S(s), Sigma(sigma), debug(0) { Layers = s + 2; K = pow(2., 1. / s); }

    // sift检测开始入口
    // 0. 参数计算 + 图像预处理
    bool detect(const cv::Mat& img, std::vector<keypoint>& kpts, cv::Mat& fvec);

    void info();

private:

    // 1. 建立金字塔 pyr_G  pyr_DoG
    void buildPyramid();

    // 2. 检测特征点
    void findFeaturePoints(std::vector<keypoint>& kpts);
    // 2.1 寻找DoG图像极值点
    // 2.2 极值点筛选
    bool filterExtrema(keypoint& kpt);
    // 2.3 计算特征点主方向
    void calcMainOrientation(keypoint& kpt, std::vector<float>& angs);

    // 3. 提取特征点处的128维特征向量
    void calcFeatureVector(std::vector<keypoint>& kpts, cv::Mat& fvec);


public:
    int debug;

private:
    int Octaves;
    int Layers; // Layers为高斯模糊金子塔一个octave内图像数-1 如 S = 3  0 1 2 3 4 5 六张图像 Layers = 6 - 1 = 5
    double Sigma;
    int S;
    double K;

    // 图像
    cv::Mat img_org; // 原图像
    cv::Mat img; // 放大两倍灰度图

    // 金子塔
    pyramid pyr_G; // 高斯模糊金字塔
    pyramid pyr_DoG; // DoG金字塔
};
