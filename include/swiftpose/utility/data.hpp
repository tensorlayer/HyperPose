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

struct feature_map_t : public ttl::tensor_ref<float, 3> {
  public:
    feature_map_t(std::string name_, std::shared_ptr<ttl::tensor<float, 4>> ptr,
                  size_t offset, size_t c, size_t h, size_t w)
        : ttl::tensor_ref<float, 3>(
              ptr->data(), ttl::tensor_ref<float, 3>::shape_type(c, h, w)),
          m_name(std::move(name_)),
          m_tensor_ptr(std::move(ptr))
    {
    }

    friend std::ostream &operator<<(std::ostream &out, const feature_map_t &map)
    {
        out << map.m_name << ":[" << map.dims()[0] << ", " << map.dims()[1]
            << ", " << map.dims()[2] << ']';
        return out;
    }

  private:
    std::shared_ptr<ttl::tensor<float, 4>> m_tensor_ptr;
    std::string m_name;
};

using internal_t = std::vector<feature_map_t>;

struct output_t {
    cv::Mat mat;
    std::vector<human_t> poses;

    cv::Mat &visualize();
    cv::Mat visualize_copied() const;
};

void images2nchw(std::vector<float> &data, std::vector<cv::Mat> images,
                 cv::Size size, double factor = 1.0, bool flip_rb = true);

}  // namespace swiftpose