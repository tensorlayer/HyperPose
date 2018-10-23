#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include <bits/std_shape.hpp>  // FIXME: don't include internal header

#include "trace.hpp"

using rank_t = ttl::internal::rank_t;
template <rank_t r> using shape = ttl::internal::basic_shape<r>;

template <typename T> struct cuda_mem_allocator {
    T *operator()(int count)
    {
        T *deviceMem;
        cudaMalloc(&deviceMem, count * sizeof(T));
        return deviceMem;
    }
};

struct cuda_mem_deleter {
    void operator()(void *ptr) { cudaFree(ptr); }
};

template <typename R, rank_t r> class basic_cuda_tensor
{
  public:
    template <typename... D>
    explicit basic_cuda_tensor(D... d)
        : shape_(d...),
          count(shape_.size()),
          data_(cuda_mem_allocator<R>()(count))
    {
        TRACE_SCOPE(__func__);
    }

    R *data() { return data_.get(); }

    void fromHost(void *buffer)
    {
        TRACE_SCOPE("basic_cuda_tensor::fromHost");
        cudaMemcpy(data_.get(), buffer, count * sizeof(R),
                   cudaMemcpyHostToDevice);
    }

    void partialFromHost(void *buffer, int batch_size, int max_batch_size)
    {
        TRACE_SCOPE("basic_cuda_tensor::partialFromHost");
        cudaMemcpy(data_.get(), buffer,
                   count / max_batch_size * batch_size * sizeof(R),
                   cudaMemcpyHostToDevice);
    }

    void toHost(void *buffer)
    {
        TRACE_SCOPE("basic_cuda_tensor::toHost");
        cudaMemcpy(buffer, data_.get(), count * sizeof(R),
                   cudaMemcpyDeviceToHost);
    }

    void partialToHost(void *buffer, int batch_size, int max_batch_size)
    {
        TRACE_SCOPE("basic_cuda_tensor::partialToHost");
        cudaMemcpy(buffer, data_.get(),
                   count / max_batch_size * batch_size * sizeof(R),
                   cudaMemcpyDeviceToHost);
    }

  private:
    const shape<r> shape_;
    const size_t count;
    std::unique_ptr<R, cuda_mem_deleter> data_;
};

template <typename R, rank_t r> using cuda_tensor = basic_cuda_tensor<R, r>;
