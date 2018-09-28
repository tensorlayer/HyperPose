#pragma once
#include <string>
#include <vector>

class stream_detector
{
  public:
    virtual ~stream_detector() {}

    virtual void run(const std::vector<std::string> &) = 0;

    static stream_detector *create(const std::string &model_file,
                                   int input_height, int input_width,  //
                                   int feature_height, int feature_width,
                                   int batch_size, bool use_f16,
                                   int gauss_kernel_size);
};