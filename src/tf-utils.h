#pragma once

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

#include "tensor.h"

namespace tf = tensorflow;

template <typename T> tf::Tensor import4dtensor(const tensor_t<T, 4> &input)
{
    // const auto [a, b, c, d] = input.dims; // requires C++17
    const int a = input.dims[0];
    const int b = input.dims[1];
    const int c = input.dims[2];
    const int d = input.dims[3];

    // TODO: infer type from T
    tf::Tensor t(tf::DT_FLOAT, tf::TensorShape({a, b, c, d}));
    {
        int idx = 0;
        for (int i = 0; i < a; ++i) {
            for (int j = 0; j < b; ++j) {
                for (int k = 0; k < c; ++k) {
                    for (int l = 0; l < d; ++l) {
                        t.tensor<T, 4>()(i, j, k, l) = input.data()[idx++];
                    }
                }
            }
        }
    }
    return t;
}

template <typename T> void export4dtensor(const tf::Tensor &t, T *data)
{
    // const auto [a, b, c, d] = output.dims; // requires C++17
    const int a = t.shape().dim_size(0);
    const int b = t.shape().dim_size(1);
    const int c = t.shape().dim_size(2);
    const int d = t.shape().dim_size(3);

    const auto &tt = t.tensor<T, 4>();
    {
        int idx = 0;
        for (int i = 0; i < a; ++i) {
            for (int j = 0; j < b; ++j) {
                for (int k = 0; k < c; ++k) {
                    for (int l = 0; l < d; ++l) {
                        data[idx++] = tt(i, j, k, l);
                    }
                }
            }
        }
    }
}
