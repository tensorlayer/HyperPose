#include <cstdint>
#include <string>

#include <opencv2/opencv.hpp>

#include <ttl/range>
#include <ttl/tensor>

void _input_image(const std::string &filename,
                  const ttl::tensor_ref<uint8_t, 3> hwc_buffer)
{
    const auto img = cv::imread(filename);
    if (img.empty()) {
        // TODO: handle error
        std::cerr << "[ERROR] cv::Mat of " << filename << "is empty.\n";
        exit(1);
    }
    const auto [h, w, _3] = hwc_buffer.dims();
    cv::Mat resized_image(cv::Size(w, h), CV_8UC(3), hwc_buffer.data());
    cv::resize(img, resized_image, resized_image.size());
}

void input_image(const std::string &filename,
                 const ttl::tensor_ref<uint8_t, 3> hwc_buffer,  // [h, w, 3]
                 const ttl::tensor_ref<float, 3> chw_buffer,    // [3, h, w]
                 bool flip_rgb)
{
    _input_image(filename, hwc_buffer);
    for (auto k : ttl::range(3)) {
        for (auto i : ttl::range<1>(chw_buffer)) {
            for (auto j : ttl::range<2>(chw_buffer)) {
                chw_buffer.at(k, i, j) =
                    hwc_buffer.at(i, j, flip_rgb ? 2 - k : k) / 255.0;
            }
        }
    }
}
