#include "visual_sfm.h"
#include <string>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "we need a param for config file" << std::endl;
    }

    std::string config_file = std::string(argv[1]);

    sfm::VisualSFM::Ptr vsfm(new sfm::VisualSFM(config_file));
    assert(vsfm->Init() == true);
    vsfm->Run();

    return 0;
}