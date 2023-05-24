#include "match.h"

// 优选匹配点
std::vector<cv::DMatch> chooseGood(cv::Mat descriptor, std::vector<cv::DMatch> matches)
{
    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < descriptor.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    std::vector<cv::DMatch> goodMatches;
    for (int i = 0; i < descriptor.rows; i++)
    {
        if (matches[i].distance < 0.2 * max_dist)
            goodMatches.push_back(matches[i]);
    }
    return goodMatches;
}

// 匹配对称性检测
void symmetryTest(std::vector<cv::DMatch>& matches1, std::vector<cv::DMatch>& matches2,
                  std::vector<cv::DMatch>& symMatches)
{
    for (auto m1 : matches1)
        for (auto m2 : matches2)
        {
            // 进行匹配测试
            if (m1.queryIdx == m2.trainIdx  &&    m2.queryIdx == m1.trainIdx)
            {
                symMatches.push_back(m1);
                break;
            }
        }
}