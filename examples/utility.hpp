//
// Created by ganler on 3/8/20.
//

#pragma once

#include "vis.h"

#include <opencv2/opencv.hpp>
#include <openpose-plus/human.h>
#include <ttl/tensor>
#include <vector>

namespace pose
{
using batch_t = std::vector<cv::Mat>;

template <typename... MatType> batch_t make_batch(MatType... mats)
{
    batch_t ret;
    ret.reserve(sizeof...(mats));
    (ret.push_back(mats), ...);  // Pack fold.
    return ret;
}

struct internal_result_t {
    ttl::tensor<float, 4> paf;
    ttl::tensor<float, 4> conf;
};

struct result_t {
    cv::Mat mat;
    human_t pose;

    cv::Mat &visualize() { draw_human(mat, pose); }
    cv::Mat visualize_copied()
    {
        auto copied = mat.clone();
        draw_human(copied, pose);
    }
};

}  // namespace pose