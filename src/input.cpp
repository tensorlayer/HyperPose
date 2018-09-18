#include <iostream>

#include <opencv2/opencv.hpp>

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
        int idx = 0;
        for (int i = 0; i < target_height; ++i) {
            for (int j = 0; j < target_width; ++j) {
                const auto pix = resized_image.at<cv::Vec3b>(i, j);
                buffer[idx++] = pix[0] / 255.0;
                buffer[idx++] = pix[1] / 255.0;
                buffer[idx++] = pix[2] / 255.0;
            }
        }
    }

    return resized_image;
}
