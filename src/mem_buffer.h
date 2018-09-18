#pragma once

#include <cstdint>
#include <iostream>
#include <memory>

namespace
{
class buffer_t
{
  public:
    virtual void *data() = 0;
    virtual int64_t memSize() const = 0;
};

}  // namespace
template <typename T> class mem_buffer_t : public buffer_t
{
  public:
    mem_buffer_t(int count)
        : count_(count), data_(new uint8_t[count * sizeof(T)])
    {
        fprintf(stderr, "mem buffer created @%p with size: %lu\n", data_.get(),
                count_ * sizeof(T));
    }

    void *data() override { return data_.get(); }
    int64_t memSize() const override { return count_ * sizeof(T); }

  private:
    const int count_;
    std::unique_ptr<uint8_t[]> data_;
};
