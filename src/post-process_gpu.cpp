#include "post-process_gpu.h"

#include <cassert>

#include "cudnn.hpp"
#include "post-process.h"
#include "tracer.h"

template <typename T>
void get_gauss_kernel(int ksize, T sigma, tensor_t<T, 2> &t)
{
    TRACE(__func__);

    const T gk17[] = {0.001663, 0.004775, 0.011909, 0.025805, 0.048580,
                      0.079458, 0.112920, 0.139427, 0.149579, 0.139427,
                      0.112920, 0.079458, 0.048580, 0.025805, 0.011909,
                      0.004775, 0.001663};
    assert(ksize == 17);
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) { t.at(i, j) = gk17[i] * gk17[j]; }
    }
}

template <typename T>
void get_peak_map_gpu_tpl(const tensor_t<T, 3> &input, tensor_t<T, 3> &output)
{
    TRACE(__func__);

    using Conv = Conv_NCHW_PaddingSame<T>;
    using Pool = Pool_NCHW_PaddingSame_Max<T>;

    const int channel = input.dims[0];
    const int height = input.dims[1];
    const int width = input.dims[2];

    const int ksize = 17;

    tensor_t<T, 2> gk(nullptr, ksize, ksize);
    get_gauss_kernel<T>(ksize, 3.0, gk);

    cuda_tensor<T, 3> input_gpu(channel, height, width);
    cuda_tensor<T, 3> smoothed_gpu(channel, height, width);
    cuda_tensor<T, 3> pooled_gpu(channel, height, width);
    tensor_t<T, 3> pooled(nullptr, channel, height, width);
    cuda_tensor<T, 2> gk_gpu(ksize, ksize);

    input_gpu.fromHost(input.data());
    gk_gpu.fromHost(gk.data());

    // conv :: [n, c, h, w], [d, c, r, s] -> [n, d, h, w]
    Conv conv(channel, 1, height, width, 1, ksize, ksize);
    Pool pool(1, channel, height, width, 3, 3);

    conv(input_gpu.data(), gk_gpu.data(), smoothed_gpu.data());
    pool(smoothed_gpu.data(), pooled_gpu.data());

    smoothed_gpu.toHost(output.data());
    pooled_gpu.toHost(pooled.data());

    inplace_select_peaks(output, pooled);
}

void get_peak_map_gpu(const tensor_t<float, 3> &input,
                      tensor_t<float, 3> &output)
{
    return get_peak_map_gpu_tpl(input, output);
}
