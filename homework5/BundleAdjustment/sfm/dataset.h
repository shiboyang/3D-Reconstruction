#pragma once

#include "camera.h"
#include "common_include.h"
#include "frame.h"

namespace sfm {

/**
 * 数据集读取
 * 构造时传入配置文件路径，配置文件的dataset_dir为数据集路径
 * Init之后可获得相机和下一帧图像
 */
class Dataset {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(const std::string& dataset_path);

    /// 初始化，返回是否成功
    bool Init();

    /// create and return the next frame containing the stereo images
    Frame::Ptr NextFrame();

    /// get camera by id
    Camera::Ptr GetCamera() const {
        return camera_;
    }

   private:
    std::string dataset_path_;
    int current_image_index_ = 1;

    Camera::Ptr camera_;
};
}  // namespace 
