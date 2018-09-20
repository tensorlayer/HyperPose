#pragma once

#include <cstdint>
#include <iostream>
#include <memory>

#include <NvInfer.h>

struct buffer_info_t {
    int64_t count;
    nvinfer1::DataType dtype;
};

class cuda_buffer_t
{
  public:
    cuda_buffer_t(const buffer_info_t &);
    ~cuda_buffer_t();

    buffer_info_t info() const;
    void *data();
    int64_t memSize() const;

    void fromHost(void *buffer);
    void toHost(void *buffer);

  private:
    buffer_info_t info_;
    void *data_;
};
