

#pragma once

/// \file data.hpp
/// \brief Data types in HyperPose.

#include "human.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace hyperpose {

/// \brief The feature map tensor class.
/// \note This class extends `ttl::tensor` with names and output stream operator.
struct feature_map_t {
public:
    /// Constructor.
    /// \param name Tensor name. (Often from DNN engine graphs)
    /// \param tensor Tensor data.
    /// \param shape Shape of tensor. (no batch dimension)
    feature_map_t(std::string name, std::unique_ptr<char[]>&& tensor, std::vector<int> shape);

    /// \brief Output operator.
    /// \param out Output stream.
    /// \param map Feature map.
    /// \return The output stream.
    friend std::ostream& operator<<(std::ostream& out, const feature_map_t& map);

    ///
    /// \return Name of this feature map.
    inline const std::string& name() const { return m_name; }

    ///
    /// \return Shape of feature map. (No batch dimension).
    inline const std::vector<int>& shape() const { return m_shape; }

    ///
    /// \tparam T View type.
    /// \return Viewed data pointer.
    template <typename T>
    inline const T* view() const
    {
        return reinterpret_cast<T*>(m_data.get());
    }

private:
    std::string m_name;
    std::unique_ptr<char[]> m_data;
    std::vector<int> m_shape;
};

/// \brief A vector of feature maps.
/// \note Usually the output of DNN engine are a list/vector of tensors(e.g., paf & conf in OpenPose).
using internal_t = std::vector<feature_map_t>;

/// \brief Batching function.
/// \param data Data vector to be appended.
/// \param images A vector of images to be batched.
/// \param factor Each element in images will multiply factor.
/// \param flip_rb Flip the BGR order to RBG or not.
/// \warning Users must ensure that the size in parameter `images` are the same.
void nhwc_images_append_nchw_batch(
    std::vector<float>& data, std::vector<cv::Mat> images, double factor = 1.0, bool flip_rb = true);

cv::Mat non_scaling_resize(const cv::Mat& input, const cv::Size& dstSize, const cv::Scalar bgcolor = { 0, 0, 0 });

} // namespace hyperpose
