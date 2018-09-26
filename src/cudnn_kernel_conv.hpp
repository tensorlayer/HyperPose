#pragma once
#include <cudnn.h>

#include "cudnn_traits.hpp"
#include "std_cuda_tensor.hpp"

template <typename T> class Conv_NCHW_PaddingSame
{
  public:
    Conv_NCHW_PaddingSame(int n, int c, int h, int w, int d, int r, int s)
        : handle(createHandle()),                       //
          convDesc(createConvDesc(r, s)),               //
          xDesc(createInputTensorDesc<T>(n, c, h, w)),  //
          wDesc(createFilterDesc(d, c, r, s)),          //
          yDesc(
              createOutputTensorDesc(convDesc.get(), xDesc.get(), wDesc.get())),
          algo(getAlgo(handle.get(), xDesc.get(), wDesc.get(), convDesc.get(),
                       yDesc.get())),
          workSpaceSizeInBytes(getWorkspaceSize(handle.get(), xDesc.get(),
                                                wDesc.get(), convDesc.get(),
                                                yDesc.get())),
          workSpace(workSpaceSizeInBytes)
    {
    }

    static cudnnConvolutionDescriptor_t createConvDesc(int r, int s)
    {
        const int arrayLength = 2;
        const int padA[] = {(r - 1) / 2, (s - 1) / 2};
        const int filterStrideA[] = {1, 1};
        const int dilationA[] = {1, 1};

        cudnnConvolutionDescriptor_t convDesc;
        check << cudnnCreateConvolutionDescriptor(&convDesc);
        check << cudnnSetConvolutionNdDescriptor(
            convDesc, arrayLength, padA, filterStrideA, dilationA,
            cudnn_conv_mode<void>::value, cudnn_data_type<T>::value);
        return convDesc;
    }

    static cudnnFilterDescriptor_t createFilterDesc(int k, int c, int h, int w)
    {
        cudnnFilterDescriptor_t wDesc;
        check << cudnnCreateFilterDescriptor(&wDesc);
        check << cudnnSetFilter4dDescriptor(wDesc, cudnn_data_type<T>::value,
                                            cudnn_tensor_format<NCHW>::value, k,
                                            c, h, w);
        return wDesc;
    }

    static cudnnTensorDescriptor_t
    createOutputTensorDesc(const cudnnConvolutionDescriptor_t convDesc,
                           const cudnnTensorDescriptor_t xDesc,
                           const cudnnFilterDescriptor_t wDesc)
    {
        int n, c, h, w;
        cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c,
                                              &h, &w);

        cudnnTensorDescriptor_t yDesc;
        check << cudnnCreateTensorDescriptor(&yDesc);
        check << cudnnSetTensor4dDescriptor(
            yDesc, cudnn_tensor_format<NCHW>::value, cudnn_data_type<T>::value,
            n, c, h, w);
        return yDesc;
    }

    static cudnnConvolutionFwdAlgo_t
    getAlgo(cudnnHandle_t handle, const cudnnTensorDescriptor_t xDesc,
            const cudnnFilterDescriptor_t wDesc,
            const cudnnConvolutionDescriptor_t convDesc,
            const cudnnTensorDescriptor_t yDesc)
    {
        const cudnnConvolutionFwdPreference_t preference =
            CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
        // CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
        // CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
        const size_t Mi = 1 << 20;
        const size_t memoryLimitInBytes = 512 * Mi;

        cudnnConvolutionFwdAlgo_t algo;
        check << cudnnGetConvolutionForwardAlgorithm(
            handle, xDesc, wDesc, convDesc, yDesc, preference,
            memoryLimitInBytes, &algo);
        printf("will use algo: %d for conv\n", algo);
        return algo;
    }

    size_t getWorkspaceSize(cudnnHandle_t handle,
                            const cudnnTensorDescriptor_t xDesc,
                            const cudnnFilterDescriptor_t wDesc,
                            const cudnnConvolutionDescriptor_t convDesc,
                            const cudnnTensorDescriptor_t yDesc)
    {
        size_t sizeInBytes;
        check << cudnnGetConvolutionForwardWorkspaceSize(
            handle, xDesc, wDesc, convDesc, yDesc, algo, &sizeInBytes);
        printf("%s need %lu Mi\n", __func__, sizeInBytes >> 20);
        return sizeInBytes;
    }

    void operator()(const T *x, const T *w, T *y)
    {
        const T alpha = 1;
        const T beta = 0;
        check << cudnnConvolutionForward(handle.get(), &alpha, xDesc.get(), x,
                                         wDesc.get(), w, convDesc.get(), algo,
                                         workSpace.data(), workSpaceSizeInBytes,
                                         &beta, yDesc.get(), y);
    }

  private:
    const std::unique_ptr<cudnnContext, ContextDeleter> handle;
    const std::unique_ptr<cudnnConvolutionStruct, ConvolutionDescriptorDeleter>
        convDesc;
    const std::unique_ptr<cudnnTensorStruct, TensorDescriptorDeleter> xDesc;
    const std::unique_ptr<cudnnFilterStruct, FilterDescriptorDeleter> wDesc;
    const std::unique_ptr<cudnnTensorStruct, TensorDescriptorDeleter> yDesc;

    const cudnnConvolutionFwdAlgo_t algo;
    const size_t workSpaceSizeInBytes;
    cuda_tensor<char, 1> workSpace;
};
