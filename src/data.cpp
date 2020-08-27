#include <hyperpose/utility/data.hpp>

namespace hyperpose {

feature_map_t::feature_map_t(std::string name, std::unique_ptr<char[]>&& tensor, std::vector<int> shape)
    : m_name(std::move(name))
    , m_data(std::move(tensor))
    , m_shape(std::move(shape))
{
}

std::ostream& operator<<(std::ostream& out, const feature_map_t& map)
{
    out << map.m_name << ":[";
    for (auto& s : map.m_shape)
        out << s << ", ";
    out << ']';
    return out;
}

void nhwc_images_append_nchw_batch(std::vector<float>& data, std::vector<cv::Mat> images, double factor, bool flip_rb)
{
    if (images.empty())
        return;

    const auto size = images.at(0).size();
    data.reserve(size.area() * 3 * images.size() + data.size());

    for (auto&& image : images) {
        assert(image.type() == CV_8UC3);
        assert(size == image.size());

        int iter_rows = image.rows;
        int iter_cols = image.cols;

        if (image.isContinuous()) {
            iter_cols = image.total();
            iter_rows = 1;
        }

        constexpr std::array<size_t, 3> no_swap{ 0, 1, 2 };
        constexpr std::array<size_t, 3> swap_rb{ 2, 1, 0 };
        const auto& index_ref = flip_rb ? swap_rb : no_swap;
        for (size_t c : index_ref)
            for (int i = 0; i < iter_rows; ++i) {
                const auto* line = image.ptr<cv::Vec3b>(i);
                for (int j = 0; j < iter_cols; ++j)
                    data.push_back((*line++)[c] * factor);
            }
    }
} // TODO: Parallel.

cv::Mat non_scaling_resize(const cv::Mat& input, const cv::Size& dstSize, const cv::Scalar bgcolor)
{
    cv::Mat output;

    double h1 = dstSize.width * (input.rows / (double)input.cols);
    double w2 = dstSize.height * (input.cols / (double)input.rows);

    if (h1 <= dstSize.height) {
        cv::resize(input, output, cv::Size(dstSize.width, h1));
    } else {
        cv::resize(input, output, cv::Size(w2, dstSize.height));
    }

    cv::copyMakeBorder(output, output, 0, dstSize.height - output.rows, 0, dstSize.width - output.cols, cv::BORDER_CONSTANT, bgcolor);

    return output;
}

} // namespace hyperpose