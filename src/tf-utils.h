#pragma once

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>

namespace tf = tensorflow;

template <typename T> tf::TensorShape to_tf_shape(const std::vector<T> &dims)
{
    tf::TensorShape shape;
    for (auto d : dims) { shape.AddDim(d); }
    return shape;
}

int volume(const tf::TensorShape &shape)
{
    int v = 1;
    for (auto d : shape.dim_sizes()) { v *= d; }
    return v;
}

template <typename T>
tf::Tensor import_tensor(const T *input, const std::vector<int> &dims)
{
    const auto shape = to_tf_shape(dims);
    const int n = volume(shape);
    tf::Tensor t(tf::DT_FLOAT, shape);  // TODO: infer shape from T
    T *data = t.flat<T>().data();
    for (int i = 0; i < n; ++i) { data[i] = input[i]; }
    return t;
}

template <typename T> void export_tensor(const tf::Tensor &t, T *output)
{
    const int n = volume(t.shape());
    const T *data = t.flat<T>().data();
    for (int i = 0; i < n; ++i) { output[i] = data[i]; }
}
