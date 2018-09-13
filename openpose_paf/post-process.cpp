#include <algorithm>
#include <array>
#include <cassert>

#include <opencv2/opencv.hpp>

#include "post-process.h"
#include "tensor.h"
#include "tracer.h"

template <typename T>
void smooth(const tensor_t<T, 3> &input, tensor_t<T, 3> &output, int ksize = 17)
{
    TRACE(__func__);
    const float sigma = 3.0;

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

template <typename T>
void same_max_pool_3x3(const int height, const int width,  //
                       const T *input, T *output)
{
    const auto at = [&](int i, int j) { return i * width + j; };

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            float max_val = input[at(i, j)];
            for (int dx = 0; dx < 3; ++dx) {
                for (int dy = 0; dy < 3; ++dy) {
                    const int nx = i + dx - 1;
                    const int ny = j + dy - 1;
                    if (0 <= nx && nx < height && 0 <= ny && ny < width) {
                        max_val = std::max(max_val, input[at(nx, ny)]);
                    }
                }
            }
            output[at(i, j)] = max_val;
        }
    }
}

template <typename T>
void same_max_pool_3x3(const tensor_t<T, 3> &input, tensor_t<T, 3> &output)
{
    TRACE(__func__);

    const int channel = input.dims[0];
    const int height = input.dims[1];
    const int width = input.dims[2];

    assert(channel == output.dims[0]);
    assert(height == output.dims[1]);
    assert(width == output.dims[2]);

    const int offset = height * width;
    for (int k = 0; k < channel; ++k) {
        same_max_pool_3x3(height, width, input.data() + k * offset,
                          output.data() + k * offset);
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
    const int n = smoothed.volume();
    int tot = 0;
    for (int i = 0; i < n; ++i) {
        bool is_peak = false;
        output.data()[i] = delta(smoothed.data()[i], peak.data()[i], is_peak);
        if (is_peak) { ++tot; }
    }
    return tot;
}

void get_peak_map(const tensor_t<float, 3> &input, tensor_t<float, 3> &output)
{
    TRACE(__func__);

    const int channel = input.dims[0];
    const int height = input.dims[1];
    const int width = input.dims[2];

    assert(channel == output.dims[0]);
    assert(height == output.dims[1]);
    assert(width == output.dims[2]);

    tensor_t<float, 3> smoothed(nullptr, channel, height, width);
    tensor_t<float, 3> pooled(nullptr, channel, height, width);

    smooth(input, smoothed);
    // debug("smoothed :: ", smoothed);
    // save(smoothed, "smoothed");
    same_max_pool_3x3(smoothed, pooled);
    // debug("pooled :: ", pooled);
    // save(pooled, "pooled");
    const int n = select_peak(smoothed, pooled, output);
    const int tot = channel * height * width;
    printf("%d peaks, %.4f%%\n", n, 100.0 * n / tot);
}
