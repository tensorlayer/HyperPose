#pragma once

#include "viz.hpp"

#include <opencv2/opencv.hpp>
#include <openpose-plus/human.h>
#include <ttl/tensor>
#include <vector>

namespace sp
{

template <typename... MatType> std::vector<cv::Mat> make_batch(MatType... mats)
{
    std::vector<cv::Mat> ret;
    ret.reserve(sizeof...(mats));
    (ret.push_back(mats), ...);  // Pack fold.
    return ret;
}


struct inference_output_node : std::pair<std::string, ttl::tensor<float, 4>>
{
    using std::pair<std::string, ttl::tensor<float, 4>>::pair;

    inline std::string& name() {
        return first;
    }

    inline const std::string& name() const {
        return first;
    }

    inline ttl::tensor<float, 4>& tensor() {
        return second;
    }

    inline const ttl::tensor<float, 4>& tensor() const {
        return second;
    }

};

using inference_result_t = std::vector<inference_output_node>;

struct result_t {
    cv::Mat mat;
    human_t pose;

    cv::Mat &visualize() { draw_human(mat, pose); }
    cv::Mat visualize_copied() const;
};

void images2nchw(
        std::vector<float> data,
        std::vector<cv::Mat> images,
        cv::Size size,
        double factor=1.0,
        bool flip_rb = true);

}  // namespace sp