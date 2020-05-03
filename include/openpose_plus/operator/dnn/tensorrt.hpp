#pragma once

/// \file tensorrt.hpp
/// \brief The DNN engine implementation of TensorRT.
/// \author Jiawei Liu(github.com/ganler)

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <future>

#include <ttl/cuda_tensor>

#include "../../utility/data.hpp"

namespace poseplus {

/// Element data type.
using data_type = nvinfer1::DataType;

namespace dnn {
    /// `tensorrt` is a class using TensorRT DNN engine to perform neural network inference.
    class tensorrt {
    public:
        /// \brief The constructor of TensorRT engine.
        ///
        /// \note Currently, we support the [`.uff`](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/uff/uff.html) files which users should
        /// specify the input and output nodes by indicating their names(the names can be inferred when
        /// converting models to `.uff` format).
        ///
        /// \param model_path Path to the model file. (currently, only `.uff` is supported)
        /// \param input_size The input size(width first, and then height) of the model.
        /// \param input_name The name of input node. (for simplicity, we only consider 1 input node model)
        /// \param output_names The names of output nodes. (e.g., openpose model will generate "paf" and "conf")
        /// \param max_batch_size The maximum batch size of the inputs. (for input/output buffer allocation)
        /// \param dtype The data type of data element. (for some GPUs, low precision data type will be faster)
        /// \param factor For each element in the input data, they will be multiplied by "factor".
        /// \param flip_rgb Whether to convert the color channels from "BGR" to "RGB".
        explicit tensorrt(const std::string& model_path, cv::Size input_size,
            const std::string& input_name,
            const std::vector<std::string>& output_names,
            int max_batch_size = 8,
            nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT,
            double factor = 1. / 255, bool flip_rgb = true);

        /// Deconstructor of class poseplus::dnn::tensorrt.
        ~tensorrt();

        ///
        /// \return The maximum batch size of this engine.
        inline int max_batch_size() noexcept { return m_max_batch_size; }

        ///
        /// \return The input `(height, width)` of this engine.
        inline cv::Size input_size() noexcept { return m_inp_size; }

        /// Do inference with `cv::Mat`(OpenCV image/matrix data structure).
        /**
         * @code
         * namespace pp = poseplus;
         *
         * // Create engine.
         * pp::dnn::tensorrt engine(...);
         *
         * // Prepare the input data.
         * auto mat = cv::imread("/path/to/images");
         *
         * // Inference 4 images.
         * engine.inference({mat, mat, mat, mat});
         *
         * @endcode
         */
        /// \param inputs A vector of inputs.
        /// \pre `inputs.size() <= max_batch_size()`(or `std::logic_error` will be thrown).
        /// \throw std::logic_error
        /// \return A vector of output feature maps(tensors).
        std::vector<internal_t> inference(std::vector<cv::Mat> inputs);

        /// \brief Do inference using plain float buffers(NCHW format required).
        /// \details This step will not involve in any scalar multiplication or channel swapping(related to the `factor` and `flip_rgb` parameter in the constructor).
        ///
        /// \see poseplus::nhwc_images_append_nchw_batch
        /// \see poseplus::tensorrt::tensorrt
        ///
        /// \param float_buffer The input float buffers.
        /// \param batch_size The batch size of inputs to do inference.
        /// \return  vector of output feature maps(tensors).
        std::vector<internal_t> inference(const std::vector<float>& float_buffer, size_t batch_size);

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