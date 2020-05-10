#pragma once

#include "human.hpp"

#include <opencv2/opencv.hpp>
#include <ttl/tensor>
#include <vector>

namespace poseplus {

template <typename... MatType>
std::vector<cv::Mat> make_batch(MatType... mats)
{
    std::vector<cv::Mat> ret;
    ret.reserve(sizeof...(mats));
    (ret.push_back(mats), ...); // Pack fold.
    return ret;
}

struct feature_map_t {
public:
    feature_map_t(std::string name, ttl::tensor<float, 3> tensor)
        : m_name(std::move(name))
        , m_tensor(std::move(tensor))
    {
    }

    friend std::ostream& operator<<(std::ostream& out, const feature_map_t& map)
    {
        const auto [a, b, c] = map.m_tensor.dims();
        out << map.m_name << ":[" << a << ", " << b << ", " << c << ']';
        return out;
    }

    inline const std::string& name() { return m_name; }

    ttl::tensor_view<float, 3> view() const
    {
        return ttl::view(m_tensor);
    }

private:
    std::string m_name;
    ttl::tensor<float, 3> m_tensor;
};

using internal_t = std::vector<feature_map_t>;

struct output_t {
    cv::Mat mat;
    std::vector<human_t> poses;

    cv::Mat& visualize();
    cv::Mat visualize_copied() const;
};

void nhwc_images_append_nchw_batch(
    std::vector<float>& data, std::vector<cv::Mat> images,
    cv::Size size, double factor = 1.0, bool flip_rb = true);

} // namespace poseplus
