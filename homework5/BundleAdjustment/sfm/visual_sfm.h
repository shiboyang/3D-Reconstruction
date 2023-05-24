#pragma once

#include "backend.h"
#include "common_include.h"
#include "dataset.h"
#include "frontend.h"
#include "viewer.h"

namespace sfm {

/**
 * VSFM 对外接口
 */
class VisualSFM {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<VisualSFM> Ptr;

    /// constructor with config file
    VisualSFM(std::string &config_path);

    /**
     * do initialization things before run
     * @return true if success
     */
    bool Init();

    /**
     * start vo in the dataset
     */
    void Run();

    /**
     * Make a step forward in dataset
     */
    bool Step();

    /// 获取前端状态
    FrontendStatus GetFrontendStatus() const { return frontend_->GetStatus(); }

   private:
    bool inited_ = false;
    std::string config_file_path_;

    Frontend::Ptr frontend_ = nullptr;
    Backend::Ptr backend_ = nullptr;
    Map::Ptr map_ = nullptr;
    Viewer::Ptr viewer_ = nullptr;

    // dataset
    Dataset::Ptr dataset_ = nullptr;
};
}  // namespace
