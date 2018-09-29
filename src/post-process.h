#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "cudnn.hpp"
#include "std_cuda_tensor.hpp"
#include "tensor.h"
#include "tracer.h"

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
        const cv::Mat input_image(size, cv::DataType<T>::type, (T *)input[k]);
        cv::Mat output_image(target_size, cv::DataType<T>::type, output[k]);
        cv::resize(input_image, output_image, output_image.size(), 0, 0,
                   CV_INTER_AREA);
    }
}

template <typename T>
void smooth(const tensor_t<T, 3> &input, tensor_t<T, 3> &output, int ksize)
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
        const cv::Mat input_image(size, cv::DataType<T>::type, (T *)input[k]);
        cv::Mat output_image(size, cv::DataType<T>::type, output[k]);
        cv::GaussianBlur(input_image, output_image, cv::Size(ksize, ksize),
                         sigma);
    }
}

template <typename T>
void same_max_pool_3x3_2d(const int height, const int width,  //
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

    for (int k = 0; k < channel; ++k) {
        same_max_pool_3x3_2d(height, width, input[k], output[k]);
    }
}

template <typename T>
void inplace_select_peaks(const tensor_t<T, 3> &output,
                          const tensor_t<T, 3> &pooled)
{
    TRACE(__func__);
    const int n = output.volume();
    // TODO: use par execution policy (requires c++17 or c++20)
    std::transform(output.data(), output.data() + n, pooled.data(),
                   output.data(), [](T x, T y) { return x != y ? 0 : x; });
}

template <typename T> class get_peak_map_op
{
  public:
    get_peak_map_op(int channel, int height, int width, int ksize)
        : ksize(ksize),
          smoothed_gpu(channel, height, width),
          pooled_gpu(channel, height, width),
          pooled_cpu(nullptr, channel, height, width),
          same_max_pool_3x3_gpu(1, channel, height, width, 3, 3)
    {
    }

    void operator()(const tensor_t<T, 3> &input, tensor_t<T, 3> &output,
                    bool use_gpu)
    {
        TRACE(std::string("<") + typeid(*this).name() + ">::" + __func__);
        smooth(input, output, ksize);
        if (use_gpu) {
            TRACE("max pooling on GPU");
            smoothed_gpu.fromHost(output.data());
            same_max_pool_3x3_gpu(smoothed_gpu.data(), pooled_gpu.data());
            // cudaDeviceSynchronize();
            pooled_gpu.toHost(pooled_cpu.data());
        } else {
            TRACE("max pooling on CPU");
            same_max_pool_3x3(output, pooled_cpu);
        }
        inplace_select_peaks(output, pooled_cpu);
    }

  private:
    const int ksize;

    cuda_tensor<T, 3> smoothed_gpu;
    cuda_tensor<T, 3> pooled_gpu;
    tensor_t<T, 3> pooled_cpu;

    using Pool = Pool_NCHW_PaddingSame_Max<T>;
    Pool same_max_pool_3x3_gpu;
};
