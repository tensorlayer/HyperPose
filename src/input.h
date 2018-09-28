#pragma once
#include <cstdint>
#include <string>

#include <opencv2/opencv.hpp>

// TODO: deprecate
// input an image and resize it to target size
cv::Mat input_image(
    const std::string &filename, int target_height, int target_width,
    // size of buffer >= target_height * target_width * sizeof(float) * 3
    float *buffer);

// input an image and resize it to target size
void input_image(const std::string &filename, int target_height,
                 int target_width, uint8_t *hwc_buffer, float *chw_buffer);
