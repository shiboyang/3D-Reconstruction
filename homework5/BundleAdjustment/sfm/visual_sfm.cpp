#include "visual_sfm.h"
#include <chrono>
#include "config.h"

namespace sfm {

VisualSFM::VisualSFM(std::string &config_path)
    : config_file_path_(config_path) {}

bool VisualSFM::Init() {
    std::cout << "Start Init " << std::endl;
    // read from config file
    if (!Config::SetParameterFile(config_file_path_)) {
        return false;
    }
    
    dataset_ = Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
    dataset_ -> Init();

    // create components and links
    frontend_ = Frontend::Ptr(new Frontend);
    backend_ = Backend::Ptr(new Backend);
    map_ = Map::Ptr(new Map);
    viewer_ = Viewer::Ptr(new Viewer);

    frontend_->SetBackend(backend_);
    frontend_->SetMap(map_);
    frontend_->SetViewer(viewer_);

    auto camera_ = dataset_->GetCamera();
    frontend_->SetCameras(camera_);
    backend_->SetMap(map_);
    backend_->SetCameras(camera_);

    viewer_->SetMap(map_);

    return true;
}

void VisualSFM::Run() {
    std::cout << "SFM is running"<<std::endl;
    int idx = 0;
    while (1) {
        // 当测试两张图像BA时，请将设定：idx==2, 如果测试incremental BA, idx==4
        if(idx == 4) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        idx++;

        if (Step() == false) {
            break;
        }
    }

    backend_->Stop();
    viewer_->Close();

    std::cout << "SFM exit";
}

bool VisualSFM::Step() {
    std::cout << "SFM Step"<<std::endl;
    Frame::Ptr new_frame = dataset_->NextFrame();
    if (new_frame == nullptr) return false;

    auto t1 = std::chrono::steady_clock::now();
    bool success = frontend_->AddFrame(new_frame);
    auto t2 = std::chrono::steady_clock::now();
    auto time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Frame cost time: " << time_used.count() << " seconds.";
    return success;
}

}  // namespace myslam
