#include <opencv2/opencv.hpp>

namespace hyperpose {

using color_t = std::tuple<uint8_t, uint8_t, uint8_t>;

inline cv::Scalar to_cv_scalar(const color_t& color)
{
    // const auto [r, g, b] = color;
    const auto r = std::get<0>(color);
    const auto g = std::get<1>(color);
    const auto b = std::get<2>(color);

    return cv::Scalar(r, g, b);
}

const std::vector<color_t> coco_colors_rgb = {
    color_t{ 255, 0, 0 }, // 0
    color_t{ 255, 85, 0 }, // 1
    color_t{ 255, 170, 0 }, // 2
    color_t{ 255, 255, 0 }, // 3
    color_t{ 170, 255, 0 }, // 4
    color_t{ 85, 255, 0 }, // 5
    color_t{ 0, 255, 0 }, // 6
    color_t{ 0, 255, 85 }, // 7
    color_t{ 0, 255, 170 }, // 8
    color_t{ 0, 255, 255 }, // 9
    color_t{ 0, 170, 255 }, // 10
    color_t{ 0, 85, 255 }, // 11
    color_t{ 0, 0, 255 }, // 12
    color_t{ 85, 0, 255 }, // 13
    color_t{ 170, 0, 255 }, // 14
    color_t{ 255, 0, 255 }, // 15
    color_t{ 255, 0, 170 }, // 16
    color_t{ 255, 0, 85 }, // 17
    color_t{ 127, 127, 127 }, // *
};

template <typename T, typename S, typename F>
std::vector<T> map(const F& f, const std::vector<S>& input)
{
    std::vector<T> output;
    output.reserve(input.size());
    for (const auto& x : input) {
        output.push_back(f(x));
    }
    return output;
}

const std::vector<cv::Scalar> coco_colors = map<cv::Scalar>(to_cv_scalar, coco_colors_rgb);

}