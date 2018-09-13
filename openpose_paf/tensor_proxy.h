#pragma once

template <typename T> struct tensor_proxy_3d_ {
    const T *const data;
    const int height;
    const int width;
    const int channel;

    tensor_proxy_3d_(const T *data, int height, int width, int channel)
        : data(data), height(height), width(width), channel(channel)
    {
    }

    const T &at(int i, int j, int k) const
    {
        const int idx = (i * width + j) * channel + k;
        return data[idx];
    }
};
