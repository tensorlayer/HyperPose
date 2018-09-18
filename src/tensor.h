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

template <typename T>
void chw_from_hwc(tensor_t<T, 3> &output, const tensor_proxy_t<T, 3> &input)
{
    const int c = output.dims[0];
    const int h = output.dims[1];
    const int w = output.dims[2];
    for (int k = 0; k < c; ++k) {
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                output.at(k, i, j) = input.at(i, j, k);
            }
        }
    }
}

template <typename T> void chw_from_hwc(tensor_t<T, 3> &output, const T *input)
{
    const int c = output.dims[0];
    const int h = output.dims[1];
    const int w = output.dims[2];
    tensor_proxy_t<float, 3> tmp((T *)input, h, w, c);
    chw_from_hwc(output, tmp);
}

// debug functions

#include <iostream>

template <typename T, uint8_t r>
void debug(const std::string &prefix, const tensor_t<T, r> &t)
{
    const auto n = volume<r>(t.dims);
    T min = *std::min_element(t.data(), t.data() + n);
    T max = *std::max_element(t.data(), t.data() + n);
    T sum = std::accumulate(t.data(), t.data() + n, (T)0);

    std::cout << "[debug] " << prefix << " :: " << show_dim<r>(t.dims) << " ["
              << min << "," << max << "] ~ " << sum / n << std::endl;
}

template <typename T>
void amax3(const tensor_t<T, 3> &images, tensor_t<T, 2> &image)
{
    const int h = images.dims[0];
    const int w = images.dims[1];
    const int c = images.dims[2];
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            T max_val = images.at(i, j, 0);
            for (int k = 1; k < c; ++k) {
                max_val = std::max(max_val, images.at(i, j, k));
            }
            image.at(i, j) = max_val;
        }
    }
}

#include <opencv2/opencv.hpp>

template <typename T>
void draw(const tensor_t<T, 2> &image, const std::string &name)
{
    const int h = image.dims[0];
    const int w = image.dims[1];
    cv::Mat img(cv::Size(w, h), CV_32F, image.data());
    cv::Mat normalized(cv::Size(w, h), CV_32F);
    cv::normalize(img, normalized, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);
    cv::imwrite(name, normalized);
}

template <typename T>
void save(const tensor_t<T, 3> &image, const std::string &name)
{
    TRACE(__func__);

    const int h = image.dims[0];
    const int w = image.dims[1];
    const int c = image.dims[2];

    for (int k = 0; k < c; ++k) {
        cv::Mat img(cv::Size(w, h), CV_32F);
        for (int i = 0; i < h; ++i) {
            for (int j = 0; j < w; ++j) {
                img.at<float>(i, j) = image.at(i, j, k);
            }
        }
        cv::Mat normalized(cv::Size(w, h), CV_32F);
        cv::normalize(img, normalized, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);
        cv::imwrite(name + "-" + std::to_string(k) + ".png", normalized);
    }
}

inline void _reverse_byte_order(uint32_t &x)
{
    x = (x << 24) | ((x << 8) & 0xff0000) | ((x >> 8) & 0xff00) | (x >> 24);
}

template <typename T, uint8_t r>
void load_idx_file(const tensor_t<T, r> &t, const std::string &filename)
{
    TRACE(__func__);

    FILE *fp = std::fopen(filename.c_str(), "r");
    if (fp == nullptr) {
        fprintf(stderr, "file NOT Found: %s\n", filename.c_str());
        exit(1);
    }

    uint8_t magic[4];
    std::fread(&magic, 4, 1, fp);  // [0, 0, dtype, rank]
    const uint8_t dtype = magic[2];
    const uint8_t rank = magic[3];
    if (dtype != 0x0d) { exit(1); }
    if (rank != r) { exit(1); }

    std::vector<uint32_t> dims(rank);
    for (auto i = 0; i < rank; ++i) {
        std::fread(&dims[i], 4, 1, fp);
        _reverse_byte_order(dims[i]);
        if (dims[i] != t.dims[i]) { exit(1); }
    }
    std::fread(t.data(), sizeof(T), volume<r>(t.dims), fp);
    std::fclose(fp);
}

template <typename T>
std::unique_ptr<tensor_t<T, 3>> load_3d_tensor(const std::string &filename)
{
    TRACE(__func__);

    FILE *fp = std::fopen(filename.c_str(), "r");
    if (fp == nullptr) {
        fprintf(stderr, "file NOT Found: %s\n", filename.c_str());
        exit(1);
    }

    uint8_t magic[4];
    std::fread(&magic, 4, 1, fp);  // [0, 0, dtype, rank]
    if (magic[2] != 0x0d) {
        fprintf(stderr, "only float is supported, want: %u, got %u", 0x0d,
                magic[2]);
        exit(1);
    }
    if (magic[3] != 3) {
        fprintf(stderr, "invalid rank, want 3, got %u\n", magic[3]);
        exit(1);
    }
    uint32_t dims[3];
    std::fread(&dims, 4, 3, fp);
    for (int i = 0; i < 3; ++i) { _reverse_byte_order(dims[i]); }
    std::unique_ptr<tensor_t<float, 3>> p(
        new tensor_t<float, 3>(nullptr, dims[0], dims[1], dims[2]));

    load_idx_file(*p, filename);
    return p;
}
