#pragma once
#include <string>
#include <vector>

class pose_detector
{
  public:
    virtual void inference(const std::vector<std::string> &image_files) = 0;

    virtual ~pose_detector() {}
};

pose_detector *create_pose_detector(const std::string &model_file,      //
                                    int input_height, int input_width,  //
                                    int feature_height, int feature_width,
                                    int batch_size, bool use_f16,
                                    int gauss_kernel_size, bool flip_rgb);
