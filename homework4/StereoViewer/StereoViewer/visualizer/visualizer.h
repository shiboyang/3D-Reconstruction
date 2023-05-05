#include <pangolin/pangolin.h>

#include <Eigen/Core>

#include "camera.h"

void DrawFrames(std::vector<Eigen::Matrix4d> PoseTwc);

void DrawModel(Points ModelPoints, Lines ModelLines);

void DrawRawFrame(Eigen::Matrix4d Twc);

void DrawEstFrame(Eigen::Matrix4d Twc);

void DrawEstimatedModel(std::vector<Eigen::Vector3d> Points);