#pragma once
#include <cassert>

#include <cuda_runtime.h>
#include <npp.h>

#include "cudnn.hpp"
#include "npp.hpp"

#include "post-process.h"
#include "std_cuda_tensor.hpp"
#include "tensor.h"
#include "tracer.h"

template <typename T> class get_peak_map_gpu_op_impl
{
  public:
    get_peak_map_gpu_op_impl(int channel, int height, int width, int ksize)
        : ksize(ksize),
          smoothed_gpu(channel, height, width),
          pooled_gpu(channel, height, width),
          pooled_cpu(nullptr, channel, height, width),
          pool(1, channel, height, width, 3, 3)
    {
    }

    void operator()(const tensor_t<T, 3> &input, tensor_t<T, 3> &output)
    {
        TRACE(std::string("<") + typeid(*this).name() + ">::" + __func__);
        smooth(input, output, ksize);
        {
            TRACE("max pooling on GPU");
            smoothed_gpu.fromHost(output.data());
            pool(smoothed_gpu.data(), pooled_gpu.data());
            // cudaDeviceSynchronize();
            pooled_gpu.toHost(pooled_cpu.data());
        }
        inplace_select_peaks(output, pooled_cpu);
    }

  private:
    const int ksize;

    cuda_tensor<T, 3> smoothed_gpu;
    cuda_tensor<T, 3> pooled_gpu;
    tensor_t<T, 3> pooled_cpu;

    using Pool = Pool_NCHW_PaddingSame_Max<T>;
    Pool pool;
};

template <typename T> class resize_op;

template <> class resize_op<float>
{
  public:
    resize_op(const int height, const int width, const int target_height,
              const int target_width, int max_batch_size)
        : height(height),
          width(width),
          target_height(target_height),
          target_width(target_width),
          max_batch_size(max_batch_size),
          oSmallestSrcSize({width, height}),
          oSrcRectROI({0, 0, width, height}),
          oSmallestDstSize({target_height, target_height}),
          oDstRectROI({0, 0, target_width, target_height}),
          eInterpolation(NPPI_INTER_CUBIC),
          batch_list_gpu(max_batch_size),
          d_input(max_batch_size, height, width),
          d_output(max_batch_size, target_height, target_width)
    {
        TRACE("resize_op::operator()");
        printf("%d x %d -> %d x %d | %d\n", height, width, target_height,
               target_width, max_batch_size);

        NppiResizeBatchCXR batch_list_cpu[max_batch_size];
        for (int i = 0; i < max_batch_size; ++i) {
            batch_list_cpu[i] = {d_input[i], width, d_output[i], target_width};
            NppiResizeBatchCXR t = batch_list_cpu[i];
            printf("%p ~ %d -> %p ~ %d\n", t.pSrc, t.nSrcStep, t.pDst,
                   t.nDstStep);
        }
        cuda_tensor<NppiResizeBatchCXR, 1> batch_list_gpu(max_batch_size);
        batch_list_gpu.fromHost(batch_list_cpu);
    }

    using T = float;

    void operator()(const T *input, T *output, int batch_size)
    {
        TRACE("resize_op::operator()");
        printf("batch_size: %d\n", batch_size);
        assert(batch_size <= max_batch_size);

        const int area = height * width;
        const int target_area = target_height * target_width;

        {
            TRACE("nppiResizeBatch_32f_C1R");
            d_input.fromHost(input, batch_size * area);
            check_npp_status << nppiResizeBatch_32f_C1R(
                oSmallestSrcSize, oSrcRectROI, oSmallestDstSize, oDstRectROI,
                eInterpolation, batch_list_gpu.data(), batch_size);
            d_output.toHost(output, batch_size * target_area);
        }
    }

  private:
    const int height;
    const int width;
    const int target_height;
    const int target_width;
    const int max_batch_size;

    const NppiSize oSmallestSrcSize;
    const NppiRect oSrcRectROI;
    const NppiSize oSmallestDstSize;
    const NppiRect oDstRectROI;
    const int eInterpolation;

    cuda_tensor<NppiResizeBatchCXR, 1> batch_list_gpu;

    cuda_tensor<T, 3> d_input;
    cuda_tensor<T, 3> d_output;
};
