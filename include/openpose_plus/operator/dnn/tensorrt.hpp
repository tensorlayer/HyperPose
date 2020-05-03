#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <future>

#include <ttl/cuda_tensor>

#include "../../utility/data.hpp"

namespace poseplus {

using data_type = nvinfer1::DataType;

namespace dnn {

    class tensorrt {
    public:
        explicit tensorrt(const std::string& model_path, cv::Size input_size,
            const std::string& input_name,
            const std::vector<std::string>& output_names,
            int max_batch_size = 8,
            nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT,
            double factor = 1. / 255, bool flip_rgb = true);

        ~tensorrt();

        inline int max_batch_size() noexcept { return m_max_batch_size; }

        inline cv::Size input_size() noexcept { return m_inp_size; }

        std::vector<internal_t> inference(std::vector<cv::Mat>);
        std::vector<internal_t> inference(const std::vector<float>&, size_t batch_size);

    private:
        const cv::Size m_inp_size; // w, h
        const int m_max_batch_size;
        const double m_factor;
        const bool m_flip_rgb;

        // Cuda related.
        struct tensorrt_deleter {
            void operator()(nvinfer1::ICudaEngine* ptr) { ptr->destroy(); }
        };
        std::unique_ptr<nvinfer1::ICudaEngine, tensorrt_deleter> m_engine;
        using cuda_buffer_t = ttl::cuda_tensor<char, 2>; // [batch_size, data_size]
        std::vector<cuda_buffer_t> m_cuda_buffers;

    private:
        void _batching(std::vector<cv::Mat>&, std::vector<float>&);
    };

} // namespace dnn

} // namespace poseplus