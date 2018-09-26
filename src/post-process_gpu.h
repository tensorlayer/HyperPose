#pragma once
#include <cassert>

#include <cuda_runtime.h>

#include "cudnn.hpp"
#include "post-process.h"
#include "std_cuda_tensor.hpp"
#include "tensor.h"
#include "tracer.h"

template <typename T> class get_peak_map_gpu_op_impl
{
  public:
    get_peak_map_gpu_op_impl(int channel, int height, int width, int ksize = 17)
        : smoothed_gpu(channel, height, width),
          pooled_gpu(channel, height, width),
          pooled_cpu(nullptr, channel, height, width),
          pool(1, channel, height, width, 3, 3)
    {
    }

    void operator()(const tensor_t<T, 3> &input, tensor_t<T, 3> &output)
    {
        TRACE(std::string("<") + typeid(*this).name() + ">::" + __func__);
        smooth(input, output);
        {
            smoothed_gpu.fromHost(output.data());
            pool(smoothed_gpu.data(), pooled_gpu.data());
            // cudaDeviceSynchronize();
            pooled_gpu.toHost(pooled_cpu.data());
        }
        inplace_select_peaks(output, pooled_cpu);
    }

  private:
    using Pool = Pool_NCHW_PaddingSame_Max<T>;

    cuda_tensor<T, 3> smoothed_gpu;
    cuda_tensor<T, 3> pooled_gpu;
    tensor_t<T, 3> pooled_cpu;

    Pool pool;
};
