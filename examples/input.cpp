#include <cstdint>
#include <string>

#include <opencv2/opencv.hpp>

#include <stdtensor>

using ttl::tensor_ref;

void _input_image(const std::string &filename, int target_height,
                  int target_width, uint8_t *hwc_buffer)
{
    const cv::Size new_size(target_width, target_height);
    cv::Mat resized_image(new_size, CV_8UC(3), hwc_buffer);

    const auto img = cv::imread(filename);
    if (img.empty()) {
        // TODO: handle error
        exit(1);
    }
    cv::resize(img, resized_image, resized_image.size(), 0, 0);
}

void input_image(const std::string &filename, int target_height,
                 int target_width, uint8_t *hwc_buffer, float *chw_buffer,
                 bool flip_rgb)
{
    _input_image(filename, target_height, target_width, hwc_buffer);
    tensor_ref<uint8_t, 3> s(hwc_buffer, target_height, target_width, 3);
    tensor_ref<float, 3> t(chw_buffer, 3, target_height, target_width);
    for (int k = 0; k < 3; ++k) {
        for (int i = 0; i < target_height; ++i) {
            for (int j = 0; j < target_width; ++j) {
                t.at(k, i, j) = s.at(i, j, flip_rgb ? 2 - k : k) / 255.0;
            }
        }
    }
}
