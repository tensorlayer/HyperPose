#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

#include <opencv2/opencv.hpp>

#include "post-process.h"
#include "tensor.h"
#include "tracer.h"

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

inline void inplace_select_peak(float &x, float y)
{
    if (x != y) { x = 0; }
}

void inplace_select_peaks(const tensor_t<float, 3> &output,
                          const tensor_t<float, 3> &pooled)
{
    TRACE(__func__);
    const int n = output.volume();
    for (int i = 0; i < n; ++i) {
        inplace_select_peak(output.data()[i], pooled.data()[i]);
    }
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

    tensor_t<float, 3> pooled(nullptr, channel, height, width);
    smooth(input, output);
    same_max_pool_3x3(output, pooled);
    inplace_select_peaks(output, pooled);
}
