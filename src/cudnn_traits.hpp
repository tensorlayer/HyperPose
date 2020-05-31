#pragma once
#include <cstdio>
#include <cstdlib>

#include <cudnn.h>

#include "trace.hpp"

/* status checker */

struct cudnn_status_checker {
    void operator<<(cudnnStatus_t status) const
    {
        if (status != CUDNN_STATUS_SUCCESS) {
            printf("want %d, got %d\n", CUDNN_STATUS_SUCCESS, status);
            perror("cudnn error");
            exit(1);
        }
    }
};

inline cudnn_status_checker check;

/* tensor formats */

struct NCHW;
struct NHWC;

template <typename T>
struct cudnn_tensor_format;

template <>
struct cudnn_tensor_format<NCHW> {
    static constexpr cudnnTensorFormat_t value = CUDNN_TENSOR_NCHW;
};

template <>
struct cudnn_tensor_format<NHWC> {
    static constexpr cudnnTensorFormat_t value = CUDNN_TENSOR_NHWC;
};

// TODO: add as needed

/* data types */

template <typename T>
struct cudnn_data_type;

template <>
struct cudnn_data_type<float> {
    static constexpr cudnnDataType_t value = CUDNN_DATA_FLOAT;
};

template <>
struct cudnn_data_type<double> {
    static constexpr cudnnDataType_t value = CUDNN_DATA_DOUBLE;
};

// TODO: add as needed

/* convolution modes */

struct CROSS_CORRELATION;

template <typename T>
struct cudnn_conv_mode;

template <>
struct cudnn_conv_mode<void> {
    static constexpr cudnnConvolutionMode_t value = CUDNN_CONVOLUTION;
};

template <>
struct cudnn_conv_mode<CROSS_CORRELATION> {
    static constexpr cudnnConvolutionMode_t value = CUDNN_CROSS_CORRELATION;
};

/* data deleters */

struct ContextDeleter {
    void operator()(cudnnHandle_t h) { check << cudnnDestroy(h); }
};

struct TensorDescriptorDeleter {
    void operator()(cudnnTensorDescriptor_t h) const
    {
        check << cudnnDestroyTensorDescriptor(h);
    }
};

struct FilterDescriptorDeleter {
    void operator()(cudnnFilterDescriptor_t h) const
    {
        check << cudnnDestroyFilterDescriptor(h);
    }
};

struct ConvolutionDescriptorDeleter {
    void operator()(cudnnConvolutionDescriptor_t h) const
    {
        check << cudnnDestroyConvolutionDescriptor(h);
    }
};

struct PoolingDescriptorDeleter {
    void operator()(cudnnPoolingDescriptor_t h) const
    {
        check << cudnnDestroyPoolingDescriptor(h);
    }
};

// TODO: add as needed

/* creators */

inline cudnnHandle_t createHandle()
{
    TRACE_SCOPE(__func__);

    cudnnHandle_t handle;
    check << cudnnCreate(&handle);
    return handle;
}

template <typename T>
cudnnTensorDescriptor_t createInputTensorDesc(int n, int c, int h, int w)
{
    cudnnTensorDescriptor_t xDesc;
    check << cudnnCreateTensorDescriptor(&xDesc);
    check << cudnnSetTensor4dDescriptor(xDesc, cudnn_tensor_format<NCHW>::value,
        cudnn_data_type<T>::value, n, c, h, w);
    return xDesc;
}

// TODO: add as needed

/* debug functions */

inline void show(const cudnnTensorDescriptor_t desc, const char* name)
{
    cudnnDataType_t dataType;
    int n, c, h, w;
    int nStride, cStride, hStride, wStride;
    check << cudnnGetTensor4dDescriptor(desc, &dataType, &n, &c, &h, &w, &nStride,
        &cStride, &hStride, &wStride);
    printf("%s :: T@<%d>[%d, %d, %d, %d] strides (%d,%d,%d,%d)\n", name, dataType,
        n, c, h, w, nStride, cStride, hStride, wStride);
}
