#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <ttl/cuda_tensor>
#include <ttl/experimental/copy>
#include <ttl/range>
#include <ttl/tensor>

#include <hyperpose/utility/human.hpp>

#include "cudnn.hpp"
#include "logging.hpp"
#include "trace.hpp"

namespace hyperpose {

// tf.image.resize_area
// This is the same as OpenCV's INTER_AREA.
// input, output are in [channel, height, width] format
template <typename T>
void resize_area(const ttl::tensor_view<T, 3>& input, const ttl::tensor_ref<T, 3>& output)
{
    TRACE_SCOPE(__func__);

    if (input.dims() == output.dims())
        return;

    const auto [channel, height, width] = input.dims();
    const auto [target_channel, target_height, target_width] = output.dims();

    assert(channel == target_channel);

    const cv::Size size(width, height);
    const cv::Size target_size(target_width, target_height);

    // What if blob => Image => Resize => Blob?
    //    cv::Mat feature_map(size, cv::DataType<T>::type, );
    // TODO: Optimize here. (50% runtime cost in PAF as the channel size is too
    // big(38)). Back soon when I get up.

    for (auto k : ttl::range(channel)) {
        const cv::Mat input_image(size, cv::DataType<T>::type, (T*)input[k].data());
        cv::Mat output_image(target_size, cv::DataType<T>::type, output[k].data());
        cv::resize(input_image, output_image, output_image.size(), 0, 0, cv::INTER_AREA);
    }
}

template <typename T>
void smooth(const ttl::tensor_view<T, 3>& input,
    const ttl::tensor_ref<T, 3>& output, int ksize)
{
    constexpr T sigma = 3.0;
    const auto [channel, height, width] = input.dims();
    const cv::Size size(width, height);
    for (auto k : ttl::range(channel)) {
        const cv::Mat input_image(size, cv::DataType<T>::type,
            (T*)input[k].data());
        cv::Mat output_image(size, cv::DataType<T>::type, output[k].data());
        if (ksize > 1)
            cv::GaussianBlur(input_image, output_image, cv::Size(ksize, ksize),
                sigma);
    }
}

template <typename T>
void same_max_pool_3x3_2d(const int height, const int width, //
    const T* input, T* output)
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
void same_max_pool_3x3(const ttl::tensor_view<T, 3>& input,
    const ttl::tensor_ref<T, 3>& output)
{
    const auto [channel, height, width] = input.dims();
    for (auto k : ttl::range(channel)) {
        same_max_pool_3x3_2d(height, width, input[k].data(), output[k].data());
    }
}

template <typename T>
T sqr(T x) { return x * x; }

template <typename T>
struct point_2d {
    T x;
    T y;

    point_2d<T> operator-(const point_2d<T>& p) const
    {
        return point_2d<T>{ x - p.x, y - p.y };
    }

    template <typename S>
    point_2d<S> cast_to() const
    {
        return point_2d<S>{ S(x), S(y) };
    }

    T l2() const { return sqr(x) + sqr(y); }
};

struct peak_info {
    int part_id;
    point_2d<int> pos;
    float score;
    int id;
};

template <typename T>
class peak_finder_t {
public:
    peak_finder_t(int channel, int height, int width, int ksize)
        : channel(channel)
        , height(height)
        , width(width)
        , ksize(ksize)
        , smoothed_cpu(channel, height, width)
        , pooled_cpu(channel, height, width)
        , same_max_pool_3x3_gpu(1, channel, height, width, 3, 3)
    {
    }

    std::vector<peak_info> find_peak_coords(const ttl::tensor_view<T, 3>& heatmap,
        float threshold, bool use_gpu)
    {
        TRACE_SCOPE(__func__);

        {
            TRACE_SCOPE("find_peak_coords::smooth");
            smooth(heatmap, ttl::ref(smoothed_cpu), ksize);
        }

        if (use_gpu) {
            TRACE_SCOPE("find_peak_coords::max pooling on GPU");
            ttl::cuda_tensor<T, 3> pool_input_gpu(channel, height, width),
                pooled_gpu(channel, height, width);
            ttl::copy(ttl::ref(pool_input_gpu), ttl::view(smoothed_cpu));
            // FIXME: pass ttl::tensor_{ref/view}
            same_max_pool_3x3_gpu(pool_input_gpu.data(), pooled_gpu.data());
            // cudaDeviceSynchronize();
            ttl::copy(ttl::ref(pooled_cpu), ttl::view(pooled_gpu));
        } else {
            TRACE_SCOPE("find_peak_coords::max pooling on CPU");
            same_max_pool_3x3(ttl::view(smoothed_cpu), ttl::ref(pooled_cpu));
        }

        std::vector<peak_info> all_peaks;
        {
            TRACE_SCOPE("find_peak_coords::find all peaks");

            const auto maybe_add_peak_info = [&](int k, int i, int j, int off) {
                if (k < COCO_N_PARTS && //
                    smoothed_cpu.data()[off] > threshold && smoothed_cpu.data()[off] == pooled_cpu.data()[off]) {
                    const int idx = all_peaks.size();
                    all_peaks.push_back(
                        peak_info{ k, point_2d<int>{ j, i }, heatmap.data()[off], idx });
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
        return all_peaks;
    }

    std::vector<std::vector<int>>
    group_by(const std::vector<peak_info>& all_peaks)
    {
        std::vector<std::vector<int>> peak_ids_by_channel(COCO_N_PARTS);
        for (const auto& pi : all_peaks) {
            peak_ids_by_channel[pi.part_id].push_back(pi.id);
        }
        return peak_ids_by_channel;
    }

    const int ksize;

private:
    const int channel;
    const int height;
    const int width;

    ttl::tensor<T, 3> smoothed_cpu;
    ttl::tensor<T, 3> pooled_cpu;

    Pool_NCHW_PaddingSame_Max<T> same_max_pool_3x3_gpu;
};
}
