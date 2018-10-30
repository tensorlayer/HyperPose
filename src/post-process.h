#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdtensor>

using ttl::tensor;
using ttl::tensor_ref;

#include <openpose-plus.h>

#include "cudnn.hpp"
#include "std_cuda_tensor.hpp"
#include "trace.hpp"

// tf.image.resize_area
// This is the same as OpenCV's INTER_AREA.
// input, output are in [channel, height, width] format
template <typename T>
void resize_area(const tensor_ref<T, 3> &input, tensor<T, 3> &output)
{
    TRACE_SCOPE(__func__);

    const int channel = input.shape().dims[0];
    const int height = input.shape().dims[1];
    const int width = input.shape().dims[2];

    const int target_channel = output.shape().dims[0];
    const int target_height = output.shape().dims[1];
    const int target_width = output.shape().dims[2];

    assert(channel == target_channel);

    const cv::Size size(width, height);
    const cv::Size target_size(target_width, target_height);
    for (int k = 0; k < channel; ++k) {
        const cv::Mat input_image(size, cv::DataType<T>::type,
                                  (T *)input[k].data());
        cv::Mat output_image(target_size, cv::DataType<T>::type,
                             output[k].data());
        cv::resize(input_image, output_image, output_image.size(), 0, 0,
                   CV_INTER_AREA);
    }
}

template <typename T>
void smooth(const tensor<T, 3> &input, tensor<T, 3> &output, int ksize)
{
    const T sigma = 3.0;

    const int channel = input.shape().dims[0];
    const int height = input.shape().dims[1];
    const int width = input.shape().dims[2];

    assert(channel == output.shape().dims[0]);
    assert(height == output.shape().dims[1]);
    assert(width == output.shape().dims[2]);

    const cv::Size size(width, height);
    for (int k = 0; k < channel; ++k) {
        const cv::Mat input_image(size, cv::DataType<T>::type,
                                  (T *)input[k].data());
        cv::Mat output_image(size, cv::DataType<T>::type, output[k].data());
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
void same_max_pool_3x3(const tensor<T, 3> &input, tensor<T, 3> &output)
{
    const int channel = input.shape().dims[0];
    const int height = input.shape().dims[1];
    const int width = input.shape().dims[2];

    assert(channel == output.shape().dims[0]);
    assert(height == output.shape().dims[1]);
    assert(width == output.shape().dims[2]);

    for (int k = 0; k < channel; ++k) {
        same_max_pool_3x3_2d(height, width, input[k].data(), output[k].data());
    }
}

template <typename T> T sqr(T x) { return x * x; }

template <typename T> struct point_2d {
    T x;
    T y;

    point_2d<T> operator-(const point_2d<T> &p) const
    {
        return point_2d<T>{x - p.x, y - p.y};
    }

    template <typename S> point_2d<S> cast_to() const
    {
        return point_2d<S>{S(x), S(y)};
    }

    T l2() const { return sqr(x) + sqr(y); }
};

struct peak_info {
    int part_id;
    point_2d<int> pos;
    float score;
    int id;
};

template <typename T> class peak_finder_t
{
  public:
    peak_finder_t(int channel, int height, int width, int ksize)
        : channel(channel),
          height(height),
          width(width),
          ksize(ksize),
          smoothed_cpu(channel, height, width),
          pooled_cpu(channel, height, width),
          pool_input_gpu(channel, height, width),
          pooled_gpu(channel, height, width),
          same_max_pool_3x3_gpu(1, channel, height, width, 3, 3)
    {
    }

    std::vector<peak_info> find_peak_coords(const tensor<float, 3> &heatmap,
                                            float threshold, bool use_gpu)
    {
        TRACE_SCOPE(__func__);

        {
            TRACE_SCOPE("find_peak_coords::smooth");
            smooth(heatmap, smoothed_cpu, ksize);
        }

        if (use_gpu) {
            TRACE_SCOPE("find_peak_coords::max pooling on GPU");
            pool_input_gpu.fromHost(smoothed_cpu.data());
            same_max_pool_3x3_gpu(pool_input_gpu.data(), pooled_gpu.data());
            // cudaDeviceSynchronize();
            pooled_gpu.toHost(pooled_cpu.data());
        } else {
            TRACE_SCOPE("find_peak_coords::max pooling on CPU");
            same_max_pool_3x3(smoothed_cpu, pooled_cpu);
        }

        std::vector<peak_info> all_peaks;
        {
            TRACE_SCOPE("find_peak_coords::find all peaks");

            const auto maybe_add_peak_info = [&](int k, int i, int j, int off) {
                if (k < COCO_N_PARTS &&  //
                    smoothed_cpu.data()[off] > threshold &&
                    smoothed_cpu.data()[off] == pooled_cpu.data()[off]) {
                    const int idx = all_peaks.size();
                    all_peaks.push_back(peak_info{k, point_2d<int>{j, i},
                                                  heatmap.data()[off], idx});
                }
            };

            int off = 0;
            for (int k = 0; k < channel; ++k) {
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        maybe_add_peak_info(k, i, j, off);
                        ++off;
                    }
                }
            }
        }
        printf("selected %lu peaks with value > %f\n", all_peaks.size(),
               threshold);
        return all_peaks;
    }

    std::vector<std::vector<int>>
    group_by(const std::vector<peak_info> &all_peaks)
    {
        std::vector<std::vector<int>> peak_ids_by_channel(COCO_N_PARTS);
        for (const auto &pi : all_peaks) {
            peak_ids_by_channel[pi.part_id].push_back(pi.id);
        }
        return peak_ids_by_channel;
    }

  private:
    const int channel;
    const int height;
    const int width;
    const int ksize;

    tensor<T, 3> smoothed_cpu;
    tensor<T, 3> pooled_cpu;
    cuda_tensor<T, 3> pool_input_gpu;
    cuda_tensor<T, 3> pooled_gpu;

    Pool_NCHW_PaddingSame_Max<T> same_max_pool_3x3_gpu;
};
