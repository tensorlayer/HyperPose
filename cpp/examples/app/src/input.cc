#include "input.h"

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

static const int image_height = 368;
static const int image_width = 432;

std::vector<float> input_image(const char *filename)
{
    const auto img = cv::imread(filename);
    const cv::Size new_size(image_width, image_height);
    cv::Mat dst(new_size, CV_8UC(3));
    cv::resize(img, dst, dst.size(), 0, 0);

    std::vector<float> output(image_height * image_width * 3);
    int idx = 0;
    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < image_width; ++j) {
            const auto pix = dst.at<cv::Vec3b>(i, j);
            output[idx++] = pix[2] / 255.0;
            output[idx++] = pix[1] / 255.0;
            output[idx++] = pix[0] / 255.0;
        }
    }
    return output;
}
