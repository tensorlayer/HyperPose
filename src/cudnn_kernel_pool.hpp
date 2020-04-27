#pragma once
#include <memory>

#include <cudnn.h>

#include "cudnn_traits.hpp"

template <typename T>
class Pool_NCHW_PaddingSame_Max {
public:
    Pool_NCHW_PaddingSame_Max(int n, int c, int h, int w, int r, int s)
        : handle(createHandle())
        , poolDesc(createPoolDesc(r, s))
        , xDesc(createInputTensorDesc<T>(n, c, h, w))
        , yDesc(createOutputTensorDesc(poolDesc.get(), xDesc.get()))
    {
    }

    static cudnnPoolingDescriptor_t createPoolDesc(int r, int s)
    {
        const cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
        const cudnnNanPropagation_t maxpoolingNanOpt = CUDNN_NOT_PROPAGATE_NAN;

        const int nbDims = 2;
        const int windowDimA[] = { r, s };
        const int padA[] = { (r - 1) / 2, (s - 1) / 2 };
        const int strideA[] = { 1, 1 };

        cudnnPoolingDescriptor_t poolDesc;
        check << cudnnCreatePoolingDescriptor(&poolDesc);
        check << cudnnSetPoolingNdDescriptor(poolDesc, mode, maxpoolingNanOpt,
            nbDims, windowDimA, padA, strideA);
        return poolDesc;
    }

    static cudnnTensorDescriptor_t
    createOutputTensorDesc(const cudnnPoolingDescriptor_t poolDesc,
        const cudnnTensorDescriptor_t xDesc)
    {
        int n, c, h, w;
        cudnnGetPooling2dForwardOutputDim(poolDesc, xDesc, &n, &c, &h, &w);
        cudnnTensorDescriptor_t yDesc;
        check << cudnnCreateTensorDescriptor(&yDesc);
        check << cudnnSetTensor4dDescriptor(yDesc, cudnn_tensor_format<NCHW>::value,
            cudnn_data_type<T>::value, n, c, h, w);
        return yDesc;
    }

    void operator()(const T* x, T* y)
    {
        const T alpha = 1;
        const T beta = 0;
        check << cudnnPoolingForward(handle.get(), poolDesc.get(), &alpha,
            xDesc.get(), x, &beta, yDesc.get(), y);
    }

private:
    const std::unique_ptr<cudnnContext, ContextDeleter> handle;
    const std::unique_ptr<cudnnPoolingStruct, PoolingDescriptorDeleter> poolDesc;
    const std::unique_ptr<cudnnTensorStruct, TensorDescriptorDeleter> xDesc;
    const std::unique_ptr<cudnnTensorStruct, TensorDescriptorDeleter> yDesc;
};
