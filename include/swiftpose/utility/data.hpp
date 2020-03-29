#pragma once

#include "viz.hpp"

#include <opencv2/opencv.hpp>
#include <ttl/tensor>
#include <vector>

namespace swiftpose
{

template <typename... MatType> std::vector<cv::Mat> make_batch(MatType... mats)
{
    std::vector<cv::Mat> ret;
    ret.reserve(sizeof...(mats));
    (ret.push_back(mats), ...);  // Pack fold.
    return ret;
}

struct feature_map_t : public ttl::tensor_ref<float, 4> {
public:

    feature_map_t(std::string name_, std::unique_ptr<ttl::tensor<float, 4>>&& ptr) :
    ttl::tensor_ref<float, 4>(*ptr),
    m_name(std::move(name_)),
    m_tensor_ptr(std::move(ptr))
    {}

    friend std::ostream& operator << (std::ostream& out, const feature_map_t& map)  {
        out << map.m_name << ":[" << map.dims()[0] << ", " << map.dims()[1] << ", " << map.dims()[2] << ", " << map.dims()[3] << ']';
        return out;
    }
private:
    std::unique_ptr<ttl::tensor<float, 4>> m_tensor_ptr;
    std::string m_name;
};

using internal_t = std::vector<feature_map_t>;

struct result_t {
    cv::Mat mat;
    human_t pose;

    cv::Mat &visualize() { draw_human(mat, pose); }
    cv::Mat visualize_copied() const;
};

void images2nchw(std::vector<float>& data, std::vector<cv::Mat> images,
                 cv::Size size, double factor = 1.0, bool flip_rb = true);

}  // namespace swiftpose