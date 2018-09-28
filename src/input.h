#pragma once
#include <string>

#include <opencv2/opencv.hpp>

// input an image and resize it to target size
cv::Mat input_image(
    const std::string &filename, int target_height, int target_width,
    // size of buffer >= target_height * target_width * sizeof(float) * 3
    float *buffer);
