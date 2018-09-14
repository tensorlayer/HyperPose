#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace
{
template <uint8_t r> int32_t volume(const std::array<int32_t, r> &dims)
{
    int32_t v = 1;
    for (auto d : dims) { v *= d; }
    return v;
}
}  // namespace

// A simple struct for tensor
template <typename T, uint8_t r> struct tensor_t {
    std::array<int32_t, r> dims;
    std::vector<T> data_;

    template <typename... Dims>
    explicit tensor_t(void *data_ptr, const Dims... dims_)
        : dims({{static_cast<int32_t>(dims_)...}}), data_(volume<r>(dims))
    {
        static_assert(sizeof...(Dims) == r, "invalid number of dims");

        fprintf(stderr, "creating tensor\n");

        if (data_ptr) {
            fprintf(stderr, "init tensor from %p\n", data_ptr);
            std::memcpy(data_.data(), data_ptr, sizeof(T) * data_.size());
        } else {
            std::memset(data_.data(), 0, sizeof(T) * data_.size());
        }
    }

    const void *data() const { return data_.data(); }

    template <typename... I> T &at(const I... i)
    {
        static_assert(sizeof...(i) == r, "invalid numer of indexes");
        const std::array<uint32_t, r> offs{{static_cast<uint32_t>(i)...}};
        uint32_t off = 0;
        for (uint8_t i = 0; i < r; ++i) { off = off * dims[i] + offs[i]; }
        return data_[off];
    }

    template <typename... I> T at(const I... i) const
    {
        static_assert(sizeof...(i) == r, "invalid numer of indexes");
        const std::array<uint32_t, r> offs{{static_cast<uint32_t>(i)...}};
        uint32_t off = 0;
        for (uint8_t i = 0; i < r; ++i) { off = off * dims[i] + offs[i]; }
        return data_[off];
    }
};

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

#include <iostream>

template <typename T, uint8_t r>
void debug(const std::string &prefix, const tensor_t<T, r> &t)
{
    std::cout << prefix << show_dim<r>(t.dims) << std::endl;
}
