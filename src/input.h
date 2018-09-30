#pragma once
#include <cstdint>
#include <string>

#include <opencv2/opencv.hpp>

// input an image and resize it to target size
void input_image(const std::string &filename, int target_height,
                 int target_width, uint8_t *hwc_buffer, float *chw_buffer,
                 bool flip_rgb);
