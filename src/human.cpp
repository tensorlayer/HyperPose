#include "coco.hpp"
#include "color.hpp"
#include <hyperpose/utility/human.hpp>

namespace hyperpose {

void draw_human(cv::Mat& img, const human_t& human)
{
    constexpr int thickness = 2;

    // draw lines
    for (int pair_id = 0; pair_id < COCO_N_PAIRS; ++pair_id) {
        const auto& pair = COCOPAIRS[pair_id];
        const auto p1 = human.parts[pair.first];
        const auto p2 = human.parts[pair.second];
        const auto color = coco_colors[pair_id];

        if (p1.has_value && p2.has_value)
            cv::line(img, cv::Point(p1.x * img.cols, p1.y * img.rows), cv::Point(p2.x * img.cols, p2.y * img.rows), color, thickness);
    }

    // draw points
    for (int part_idx = 0; part_idx < COCO_N_PARTS; ++part_idx) {
        const auto color = coco_colors[part_idx];
        const auto p = human.parts[part_idx];
        if (p.has_value) {
            cv::circle(img, cv::Point(p.x * img.cols, p.y * img.rows), thickness, color, thickness);
        }
    }
}

} // namespace hyperpose