#pragma once
#include <opencv2/opencv.hpp>

#include "tensor.h"
#include "tracer.h"

inline int area(const cv::Size &size) { return size.height * size.width; }

// tf.image.resize_area
// This is the same as OpenCV's INTER_AREA.
// input, output are in [channel, height, width] format
template <typename T>
void resize_area(const tensor_proxy_t<T, 3> &input, tensor_t<T, 3> &output)
{
    TRACE(__func__);

    const int channel = input.dims[0];
    const int height = input.dims[1];
    const int width = input.dims[2];

    const int target_channel = output.dims[0];
    const int target_height = output.dims[1];
    const int target_width = output.dims[2];

    assert(channel == target_channel);

    const cv::Size size(width, height);
    const cv::Size target_size(target_width, target_height);
    for (int k = 0; k < channel; ++k) {
        cv::Mat input_image(size, cv::DataType<T>::type,
                            input.data() + k * area(size));
        cv::Mat output_image(target_size, cv::DataType<T>::type,
                             output.data() + k * area(target_size));
        cv::resize(input_image, output_image, output_image.size(), 0, 0,
                   CV_INTER_AREA);
    }
}

template <typename T>
void smooth(const tensor_t<T, 3> &input, tensor_t<T, 3> &output, int ksize = 17)
{
    TRACE(__func__);
    const T sigma = 3.0;

    const int channel = input.dims[0];
    const int height = input.dims[1];
    const int width = input.dims[2];

    assert(channel == output.dims[0]);
    assert(height == output.dims[1]);
    assert(width == output.dims[2]);

    const cv::Size size(width, height);
    for (int k = 0; k < channel; ++k) {
        cv::Mat input_image(size, cv::DataType<T>::type,
                            input.data() + k * area(size));
        cv::Mat output_image(size, cv::DataType<T>::type,
                             output.data() + k * area(size));
        cv::GaussianBlur(input_image, output_image, cv::Size(ksize, ksize),
                         sigma);
    }
}

void inplace_select_peaks(const tensor_t<float, 3> &output,
                          const tensor_t<float, 3> &pooled);

void get_peak_map(const tensor_t<float, 3> &input, tensor_t<float, 3> &output);
