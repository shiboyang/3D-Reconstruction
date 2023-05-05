
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include "types.h"
#include "camera.h"
#include "utils.h"
#include "essential_matrix.h"


// Compute stereo from relative points
void RectifyStereoCamerasByPoints(const Camera &camera,
                                  const std::vector<Eigen::Vector2d> &normal_points1,
                                  const std::vector<Eigen::Vector2d> &normal_points2,
                                  const std::vector<Eigen::Vector2d> &points1,
                                  const std::vector<Eigen::Vector2d> &points2,
                                  Eigen::Matrix3d *H1,
                                  Eigen::Matrix3d *H2);

Eigen::Matrix3d calculate_holograph_matrix2(const Eigen::Vector3d &e,
                                            double w,
                                            double h);

Eigen::Matrix3d calculate_holograph_matrix1(const Eigen::Matrix3d *H2,
                                            Eigen::Matrix3d &F,
                                            const Eigen::Vector3d &ep,
                                            const Eigen::Vector3d &e,
                                            const std::vector<Eigen::Vector2d> &pts1,
                                            const std::vector<Eigen::Vector2d> &pts2);


Eigen::Vector3d calculate_epipolar(const std::vector<Eigen::Vector2d> &points1,
                                   const Eigen::Matrix3d &F);

