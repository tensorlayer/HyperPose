#include <algorithm>
#include <array>
#include <cassert>

#include <opencv2/opencv.hpp>

#include "tensor.h"
#include "tracer.h"

// tf.image.resize_area
// This is the same as OpenCV's INTER_AREA.
void resize_area(const tensor_t<float, 3> &input, tensor_t<float, 3> &output)
{
    TRACE(__func__);

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    const int target_height = output.dims[0];
    const int target_width = output.dims[1];
    const int target_channel = output.dims[2];

    assert(channel == target_channel);

    cv::Mat input_image(cv::Size(width, height), cv::DataType<float>::type);
    cv::Mat output_image(cv::Size(target_width, target_height),
                         cv::DataType<float>::type);

    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                input_image.at<float>(i, j) = input.at(i, j, k);
            }
        }
        cv::resize(input_image, output_image, output_image.size(), 0, 0,
                   CV_INTER_AREA);
        for (int i = 0; i < target_height; ++i) {
            for (int j = 0; j < target_width; ++j) {
                output.at(i, j, k) = output_image.at<float>(i, j);
            }
        }
    }
}

void smooth(const tensor_t<float, 3> &input, tensor_t<float, 3> &output,
            int ksize = 25)
{
    TRACE(__func__);
    const float sigma = 3.0;

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    assert(height == output.dims[0]);
    assert(width == output.dims[1]);
    assert(channel == output.dims[2]);

    const cv::Size size(width, height);
    cv::Mat input_image(size, cv::DataType<float>::type);
    cv::Mat output_image(size, cv::DataType<float>::type);

    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                input_image.at<float>(i, j) = input.at(i, j, k);
            }
        }
        cv::GaussianBlur(input_image, output_image, cv::Size(ksize, ksize),
                         sigma);
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                output.at(i, j, k) = output_image.at<float>(i, j);
            }
        }
    }
}

void maxpool_3x3(const tensor_t<float, 3> &input, tensor_t<float, 3> &output)
{
    TRACE(__func__);

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    assert(height == output.dims[0]);
    assert(width == output.dims[1]);
    assert(channel == output.dims[2]);

    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float max_val = input.at(i, j, k);
                for (int dx = 0; dx < 3; ++dx) {
                    for (int dy = 0; dy < 3; ++dy) {
                        const int nx = i + dx - 1;
                        const int ny = j + dy - 1;
                        if (0 <= nx && nx < height && 0 <= ny && ny < width) {
                            max_val = std::max(max_val, input.at(nx, ny, k));
                        }
                    }
                }
                output.at(i, j, k) = max_val;
            }
        }
    }
}

float delta(float x, float y, bool &flag)
{
    if (x == y) {
        flag = true;
        return x;
    }
    return 0;
}

int select_peak(const tensor_t<float, 3> &smoothed,
                const tensor_t<float, 3> &peak, tensor_t<float, 3> &output)
{
    TRACE(__func__);

    const int height = smoothed.dims[0];
    const int width = smoothed.dims[1];
    const int channel = smoothed.dims[2];

    int tot = 0;
    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                bool is_peak = false;
                output.at(i, j, k) =
                    delta(smoothed.at(i, j, k), peak.at(i, j, k), is_peak);
                if (is_peak) { ++tot; }
            }
        }
    }

    return tot;
}

void get_peak(const tensor_t<float, 3> &input, tensor_t<float, 3> &output)
{
    TRACE(__func__);

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    assert(height == output.dims[0]);
    assert(width == output.dims[1]);
    assert(channel == output.dims[2]);

    tensor_t<float, 3> smoothed(nullptr, height, width, channel);
    tensor_t<float, 3> pooled(nullptr, height, width, channel);

    smooth(input, smoothed);
    // debug("smoothed :: ", smoothed);
    // save(smoothed, "smoothed");
    maxpool_3x3(smoothed, pooled);
    // debug("pooled :: ", pooled);
    // save(pooled, "pooled");
    const int n = select_peak(smoothed, pooled, output);
    const int tot = height * width * channel;
    printf("%d peaks, %.4f%%\n", n, 100.0 * n / tot);
}
