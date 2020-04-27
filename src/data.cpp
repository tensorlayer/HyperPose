#include <swiftpose/utility/data.hpp>

namespace swiftpose {

cv::Mat& output_t::visualize()
{
    for (auto&& p : poses)
        draw_human(mat, p);
    return mat;
}

cv::Mat output_t::visualize_copied() const
{
    cv::Mat mat_ = mat.clone();
    for (auto&& p : poses)
        draw_human(mat_, p);
    return mat_;
}

void nhwc_images_append_nchw_batch(std::vector<float>& data,
    std::vector<cv::Mat> images, cv::Size size,
    double factor, bool flip_rb)
{
    data.reserve(size.area() * 3 * images.size() + data.size());

    for (auto&& image : images) {
        assert(image.type() == CV_8UC3);

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

} // namespace swiftpose