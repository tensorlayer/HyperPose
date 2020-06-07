#pragma once

/// \file human.hpp
/// \brief About human topology and visualization.

#include <opencv2/opencv.hpp>

namespace hyperpose {

constexpr int COCO_N_PARTS = 18;
constexpr int COCO_N_PAIRS = 19;

/// \brief Class to describe a key point.
struct body_part_t {
    bool has_value = false; ///< Whether this key point is valid.
    float x = 0; ///< X coordinate of the key point.
    float y = 0; ///< Y coordinate of the key point.
    float score = 0; ///< The inferred score(higher means more key-point-like) of the key point.
};

/// Template class to describe a human.
/// \tparam J The maximum key point of a human.
template <size_t J>
struct human_t_ {
    std::array<body_part_t, J> parts; ///< An array to tell all key point information. The index means the position.
    float score;

    inline void print() const
    {
        for (int i = 0; i < J; ++i) {
            const auto p = parts[i];
            if (p.has_value) {
                printf("BodyPart:%d-(%.2f, %.2f) score=%.2f ", i, p.x, p.y, p.score);
            }
        }
        printf("score=%.2f\n", score);
    }
};

/// \brief Class to describe a COCO human type(18 parts and 19 pairs).
/// \see　
using human_t = human_t_<COCO_N_PARTS>;

/// Function to visualize the human topology on an image.
/// \param img Image to be visualized.
/// \param human Human topology.
void draw_human(cv::Mat& img, const human_t& human);

} // namespace hyperpose