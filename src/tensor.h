#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>

#include "tracer.h"

namespace
{
template <uint8_t r> std::string show_dim(const std::array<int32_t, r> &dims)
{
    std::string s;
    for (auto d : dims) {
        if (!s.empty()) { s += ", "; }
        s += std::to_string(d);
    }
    return "(" + s + ")";
}
}  // namespace

namespace
{
template <uint8_t r> int32_t volume(const std::array<int32_t, r> &dims)
{
    int32_t v = 1;
    for (auto d : dims) { v *= d; }
    return v;
}
}  // namespace

template <uint8_t r, typename... I>
int32_t offset(const std::array<int32_t, r> &dims, const I... i)
{
    static_assert(sizeof...(i) == r, "invalid numer of indexes");
    const std::array<int32_t, r> offs{{static_cast<int32_t>(i)...}};
    int32_t off = 0;
    for (uint8_t i = 0; i < r; ++i) {
        if (offs[i] >= dims[i]) {
            printf("out of range\n");
            exit(1);
        }
        off = off * dims[i] + offs[i];
    }
    return off;
}

// A simple struct for tensor
template <typename T, uint8_t r> struct tensor_t {
    const std::array<int32_t, r> dims;
    const std::unique_ptr<T[]> data_;

    template <typename... Dims>
    explicit tensor_t(const T *data_ptr, const Dims... dims_)
        : dims({{static_cast<int32_t>(dims_)...}}),
          data_(new T[::volume<r>(dims)])
    {
        TRACE(__func__);

        static_assert(sizeof...(Dims) == r, "invalid number of dims");

        printf("creating tensor :: %s @ %p\n", show_dim<r>(dims).c_str(),
               data_.get());

        if (data_ptr) {
            std::memcpy(data_.get(), data_ptr, sizeof(T) * volume());
        } else {
            std::memset(data_.get(), 0, sizeof(T) * volume());
        }
    }

    inline T *data() const { return data_.get(); }

    template <typename... I> T &at(const I... i)
    {
        return data_[offset<r>(dims, i...)];
    }

    template <typename... I> T at(const I... i) const
    {
        return data_[offset<r>(dims, i...)];
    }

    virtual ~tensor_t() { printf("%p freed\n", data_.get()); }

    int32_t volume() const { return ::volume<r>(dims); }
};

template <typename T, uint8_t r> struct tensor_proxy_t {
    const std::array<int32_t, r> dims;
    T *const data_;

    template <typename... Dims>
    tensor_proxy_t(T *data, const Dims... dims_)
        : dims({{static_cast<int32_t>(dims_)...}}), data_(data)
    {
    }

    inline T *data() const { return data_; }

    template <typename... I> T &at(const I... i)
    {
        return data_[offset<r>(dims, i...)];
    }
    template <typename... I> T at(const I... i) const
    {
        return data_[offset<r>(dims, i...)];
    }

    int32_t volume() const { return ::volume<r>(dims); }
};
