#include "cuda_buffer.h"

#include <sys/stat.h>

#include <cassert>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "tracer.h"

#define CHECK(status)                                                          \
    do {                                                                       \
        auto ret = (status);                                                   \
        if (ret != 0) {                                                        \
            std::cout << "Cuda failure: " << ret;                              \
            abort();                                                           \
        }                                                                      \
    } while (0)

inline unsigned int elementSize(nvinfer1::DataType t)
{
    switch (t) {
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    assert(0);
    return 0;
}

inline int64_t memSize(const buffer_info_t &info)
{
    return info.count * elementSize(info.dtype);
}

void *safeCudaMalloc(size_t memSize)
{
    TRACE(__func__);
    void *deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr) {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

cuda_buffer_t::cuda_buffer_t(const buffer_info_t &info)
    : info_(info), data_(safeCudaMalloc(::memSize(info)))
{
    fprintf(stderr, "cuda buffer created @%p with size: %lu\n", data_,
            memSize());
}

cuda_buffer_t::~cuda_buffer_t() { CHECK(cudaFree(data_)); }

buffer_info_t cuda_buffer_t::info() const { return info_; }

void *cuda_buffer_t::data() { return data_; }

int64_t cuda_buffer_t::memSize() const { return ::memSize(info_); }

void cuda_buffer_t::fromHost(void *buffer)
{
    TRACE("cuda_buffer_t::fromHost");
    fprintf(stderr, "%p@cuda <- %p@mem | size = %lu\n", data_, buffer,
            memSize());
    CHECK(cudaMemcpy(data_, buffer, memSize(), cudaMemcpyHostToDevice));
}

void cuda_buffer_t::toHost(void *buffer)
{
    TRACE("cuda_buffer_t::toHost");
    fprintf(stderr, "%p@mem <- %p@cuda | size = %lu\n", buffer, data_,
            memSize());
    CHECK(cudaMemcpy(buffer, data_, memSize(), cudaMemcpyDeviceToHost));
}
