#include <iostream>

#include <opencv2/opencv.hpp>

#include "tensor.h"

std::string show_size(const cv::Size &s)
{
    return std::to_string(s.width) + " x " + std::to_string(s.height);
}

cv::Mat input_image(const char *filename, int target_height, int target_width,
                    float *buffer)
{
    auto &debug = std::cerr;

    const cv::Size new_size(target_width, target_height);
    cv::Mat resized_image(new_size, CV_8UC(3));

    const auto img = cv::imread(filename);

    if (img.empty()) {
        debug << "failed to read " << filename << std::endl;
        exit(1);
    }

    cv::resize(img, resized_image, resized_image.size(), 0, 0);

    debug << "original input image size: " << show_size(img.size())
          << ", resized to " << show_size(resized_image.size()) << std::endl;

    if (buffer) {
        tensor_proxy_t<float, 3> t(buffer, 3, target_height, target_width);
        for (int i = 0; i < target_height; ++i) {
            for (int j = 0; j < target_width; ++j) {
                const auto pix = resized_image.at<cv::Vec3b>(i, j);
                for (int k = 0; k < 3; ++k) { t.at(k, i, j) = pix[k] / 255.0; }
            }
        }
    }

    return resized_image;
}
