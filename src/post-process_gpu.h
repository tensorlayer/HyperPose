#pragma once
#include <cassert>

#include <cuda_runtime.h>

#include "cudnn.hpp"
#include "post-process.h"
#include "tensor.h"
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

template <typename T> class get_peak_map_gpu_op_impl
{
  public:
    get_peak_map_gpu_op_impl(int channel, int height, int width, int ksize = 17)
        : input_gpu(channel, height, width),
          smoothed_gpu(channel, height, width),
          pooled_gpu(channel, height, width),
          pooled_cpu(nullptr, channel, height, width),
          gk_gpu(ksize, ksize),
          conv(channel, 1, height, width, 1, ksize, ksize),
          pool(1, channel, height, width, 3, 3)
    {
        tensor_t<T, 2> gk(nullptr, ksize, ksize);
        get_gauss_kernel<T>(ksize, 3.0, gk);
        gk_gpu.fromHost(gk.data());
    }

    void operator()(const tensor_t<T, 3> &input, tensor_t<T, 3> &output)
    {
        TRACE(std::string("<") + typeid(*this).name() + ">::" + __func__);
        {
            TRACE("input, gk to device");
            input_gpu.fromHost(input.data());
        }
        {
            TRACE("op conv,pool, with sync");
            {
                TRACE("sync before");
                cudaDeviceSynchronize();
            }
            {
                TRACE("conv::op");
                conv(input_gpu.data(), gk_gpu.data(), smoothed_gpu.data());
                cudaDeviceSynchronize();
            }
            {
                TRACE("pool::op");
                pool(smoothed_gpu.data(), pooled_gpu.data());
                cudaDeviceSynchronize();
            }
            {
                TRACE("sync after");
                cudaDeviceSynchronize();
            }
        }

        {
            TRACE("smoothed, pooled to host");
            pooled_gpu.toHost(pooled_cpu.data());
            smoothed_gpu.toHost(output.data());
        }
        inplace_select_peaks(output, pooled_cpu);
    }

  private:
    using Conv = Conv_NCHW_PaddingSame<T>;
    using Pool = Pool_NCHW_PaddingSame_Max<T>;

    cuda_tensor<T, 3> input_gpu;
    cuda_tensor<T, 3> smoothed_gpu;
    cuda_tensor<T, 3> pooled_gpu;
    tensor_t<T, 3> pooled_cpu;
    cuda_tensor<T, 2> gk_gpu;

    // conv :: [n, c, h, w], [d, c, r, s] -> [n, d, h, w]
    Conv conv;
    Pool pool;
};
