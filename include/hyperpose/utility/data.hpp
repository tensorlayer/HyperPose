#pragma once

/// \file data.hpp
/// \brief Data types in OpenPose-Plus.

#include "human.hpp"

#include <opencv2/opencv.hpp>
#include <ttl/tensor>
#include <vector>

namespace hyperpose {

/// \brief The feature map tensor class.
/// \note This class extends `ttl::tensor` with names and output stream operator.
struct feature_map_t {
public:
    /// Constructor.
    /// \param name Tensor name. (Often from DNN engine graphs)
    /// \param tensor Tensor data.
    inline feature_map_t(std::string name, ttl::tensor<float, 3> tensor)
        : m_name(std::move(name))
        , m_tensor(std::move(tensor))
    {
    }

    /// \brief Output operator.
    /// \param out Output stream.
    /// \param map Feature map.
    /// \return The output stream.
    inline friend std::ostream& operator<<(std::ostream& out, const feature_map_t& map)
    {
        const auto [a, b, c] = map.m_tensor.dims();
        out << map.m_name << ":[" << a << ", " << b << ", " << c << ']';
        return out;
    }

    ///
    /// \return Name of this feature map.
    inline const std::string& name() { return m_name; }

    ///
    /// \return Tensor view of internal `ttl::tensor`.
    inline ttl::tensor_view<float, 3> view() const
    {
        return ttl::view(m_tensor);
    }

private:
    std::string m_name;
    ttl::tensor<float, 3> m_tensor;
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

} // namespace hyperpose
