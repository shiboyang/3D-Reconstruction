#include <opencv2/opencv.hpp>
#include "sift.h"

// 优选匹配点
std::vector<cv::DMatch> chooseGood(cv::Mat descriptor, std::vector<cv::DMatch> matches);


// 匹配对称性检测
void symmetryTest(std::vector<cv::DMatch>& matches1, std::vector<cv::DMatch>& matches2,
                  std::vector<cv::DMatch>& symMatches);
