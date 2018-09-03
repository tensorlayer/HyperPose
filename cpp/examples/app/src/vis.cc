#include "vis.h"

#include <opencv2/opencv.hpp>

#include "human.h"

namespace
{
template <typename T, typename S, typename F>
std::vector<T> map(const F &f, const std::vector<S> &input)
{
    std::vector<T> output;
    for (const auto &x : input) { output.push_back(f(x)); }
    return output;
}

using color_t = std::tuple<uint8_t, uint8_t, uint8_t>;

cv::Scalar to_cv_scalar(const color_t &color)
{
    const auto [r, g, b] = color;
    return cv::Scalar(r, g, b);
}

const std::vector<color_t> coco_colors_rgb = {
    {255, 0, 0},   {255, 85, 0},  {255, 170, 0}, {255, 255, 0}, {170, 255, 0},
    {85, 255, 0},  {0, 255, 0},   {0, 255, 85},  {0, 255, 170}, {0, 255, 255},
    {0, 170, 255}, {0, 85, 255},  {0, 0, 255},   {85, 0, 255},  {170, 0, 255},
    {255, 0, 255}, {255, 0, 170}, {255, 0, 85},
};

}  // namespace

const std::vector<cv::Scalar> coco_colors =
    map<cv::Scalar>(to_cv_scalar, coco_colors_rgb);

const std::vector<std::pair<int, int>> coco_pairs = {
    {1, 2},   {1, 5},  {2, 3},   {3, 4},   {5, 6},   {6, 7}, {1, 8},
    {8, 9},   {9, 10}, {1, 11},  {11, 12}, {12, 13}, {1, 0}, {0, 14},
    {14, 16}, {0, 15}, {15, 17}, {2, 16},  {5, 17},
};

void draw_human(cv::Mat &img, const Human &human)
{
    static const int image_height = 368;
    static const int image_width = 432;

    const auto f = [&](float x, float y) {
        return cv::Point((int)(x * image_width), (int)(y * image_height));
    };

    const int thickness = 3;
    const int n_pos = 19;

    // draw lines
    {
        int idx = 0;
        for (auto &[part_idx_1, part_idx_2] : coco_pairs) {
            const auto color = coco_colors[idx++];
            const auto result_1 = human.get_part(part_idx_1);
            const auto result_2 = human.get_part(part_idx_2);
            if (result_1.has_value() && result_2.has_value()) {
                const auto part_1 = result_1.value();
                const auto part_2 = result_2.value();
                cv::line(img,                        //
                         f(part_1.x(), part_1.y()),  //
                         f(part_2.x(), part_2.y()),  //
                         color, thickness);
            }
        }
    }

    // draw points
    {
        for (int part_idx = 0; part_idx < n_pos - 1; ++part_idx) {
            const auto color = coco_colors[part_idx];
            const auto result = human.get_part(part_idx);
            if (!result.has_value()) { continue; }
            const auto part = result.value();
            cv::circle(img, f(part.x(), part.y()), thickness, color, thickness);
        }
    }
}
