#include "vis.h"

#include <opencv2/opencv.hpp>

#include <openpose-plus.h>

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
    // const auto [r, g, b] = color;
    const auto r = std::get<0>(color);
    const auto g = std::get<1>(color);
    const auto b = std::get<2>(color);

    return cv::Scalar(r, g, b);
}

const std::vector<color_t> coco_colors_rgb = {
    color_t{255, 0, 0},      // 0
    color_t{255, 85, 0},     // 1
    color_t{255, 170, 0},    // 2
    color_t{255, 255, 0},    // 3
    color_t{170, 255, 0},    // 4
    color_t{85, 255, 0},     // 5
    color_t{0, 255, 0},      // 6
    color_t{0, 255, 85},     // 7
    color_t{0, 255, 170},    // 8
    color_t{0, 255, 255},    // 9
    color_t{0, 170, 255},    // 10
    color_t{0, 85, 255},     // 11
    color_t{0, 0, 255},      // 12
    color_t{85, 0, 255},     // 13
    color_t{170, 0, 255},    // 14
    color_t{255, 0, 255},    // 15
    color_t{255, 0, 170},    // 16
    color_t{255, 0, 85},     // 17
    color_t{127, 127, 127},  // *
};

}  // namespace

const std::vector<cv::Scalar> coco_colors =
    map<cv::Scalar>(to_cv_scalar, coco_colors_rgb);

void draw_human(cv::Mat &img, const human_t &human)
{
    const int thickness = 2;

    // draw lines
    for (int pair_id = 0; pair_id < COCO_N_PAIRS; ++pair_id) {
        const auto pair = COCOPAIRS[pair_id];
        const auto p1 = human.parts[pair.first];
        const auto p2 = human.parts[pair.second];
        const auto color = coco_colors[pair_id];

        if (p1.has_value && p2.has_value) {
            cv::line(img, cv::Point(p1.x, p1.y), cv::Point(p2.x, p2.y), color,
                     thickness);
        }
    }

    // draw points
    for (int part_idx = 0; part_idx < COCO_N_PARTS; ++part_idx) {
        const auto color = coco_colors[part_idx];
        const auto p = human.parts[part_idx];
        if (p.has_value) {
            cv::circle(img, cv::Point(p.x, p.y), thickness, color, thickness);
        }
    }
}
