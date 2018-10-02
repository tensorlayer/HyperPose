#pragma once
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <openpose-plus.h>

class stream_detector
{
  public:
    struct inputer_t {
        virtual bool operator()(int height, int width, uint8_t *hwc_ptr,
                                float *chw_ptr) = 0;
    };

    struct handler_t {
        virtual void operator()(cv::Mat &image,
                                const std::vector<human_t> &humans) = 0;
    };

    virtual ~stream_detector() {}

    virtual void run(inputer_t &, handler_t &, int count) = 0;

    virtual void run(const std::vector<std::string> &) = 0;

    static stream_detector *create(const std::string &model_file,
                                   int input_height, int input_width,  //
                                   int feature_height, int feature_width,
                                   int batch_size, bool use_f16,
                                   int gauss_kernel_size, bool flip_rgb);
};
