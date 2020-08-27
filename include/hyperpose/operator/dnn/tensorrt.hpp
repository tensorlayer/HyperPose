#pragma once

/// \file tensorrt.hpp
/// \brief The DNN engine implementation of TensorRT.

#include "../../utility/model.hpp"
#include <future>

#include "../../utility/data.hpp"

namespace hyperpose {

/// Data type related to TensorRT data type.
struct data_type {
    static constexpr int kFLOAT = 0; //!< FP32 format.
    static constexpr int kHALF = 1; //!< FP16 format.
    static constexpr int kINT8 = 2; //!< quantized INT8 format.
    static constexpr int kINT32 = 3; //!< INT32 format.
    static constexpr int kBOOL = 4; //!< BOOL format.

    int val = kFLOAT;
    inline data_type(int v)
        : val(v)
    {
    }
};

/// \brief The namespace to contain things related to DNN. (e.g., DNN engines and model configurations.)
/// \note In HyperPose, the pose estimation pipeline consists of DNN inference and parsing(post-processing). The DNN
/// part implementation is under the namespace `hyperpose::dnn`.
namespace dnn {
    /// \brief `tensorrt` is a class using TensorRT DNN engine to perform neural network inference.
    class tensorrt {
    public:
        /// \brief The constructor of TensorRT engine using UFF model file.
        ///
        /// \param uff_model See `hyperpose::dnn::uff`.
        /// \param input_size The input image size(height, width).
        /// \param max_batch_size The maximum batch size of the inputs. (for input/output buffer allocation)
        /// \param keep_ratio Whether to keep original aspect ratio. This is good for accuracy, but requires extra steps to refine the `hyperpose::human_t`.
        /// \param dtype The data type of data element. (for some GPUs, low precision data type will be faster)
        /// \param factor For each element in the input data, they will be multiplied by "factor".
        /// \param flip_rgb Whether to convert the color channels from "BGR" to "RGB".
        explicit tensorrt(const uff& uff_model,
            cv::Size input_size,
            int max_batch_size = 8,
            bool keep_ratio = false,
            data_type dtype = data_type::kFLOAT,
            double factor = 1. / 255, bool flip_rgb = true);

        /// \brief The constructor of TensorRT engine using ONNX model file.
        ///
        /// \param onnx_model See `hyperpose::dnn::onnx`.
        /// \param input_size The input image size(width, height).
        /// \param max_batch_size The maximum batch size of the inputs. (for input/output buffer allocation)
        /// \param keep_ratio Whether to keep original aspect ratio. This is good for accuracy, but requires extra steps to refine the `hyperpose::human_t`.
        /// \param dtype The data type of data element. (for some GPUs, low precision data type will be faster)
        /// \param factor For each element in the input data, they will be multiplied by "factor".
        /// \param flip_rgb Whether to convert the color channels from "BGR" to "RGB".
        explicit tensorrt(const onnx& onnx_model, cv::Size input_size, int max_batch_size = 8, bool keep_ratio = false,
            data_type dtype = data_type::kFLOAT,
            double factor = 1. / 255, bool flip_rgb = true);

        /// \brief The constructor of TensorRT engine using TensorRT serialized model file.
        ///
        /// \param serialized_model See `hyperpose::dnn::tensorrt_serialized`.
        /// \param input_size The input image size(width, height).
        /// \param max_batch_size The maximum batch size of the inputs. (for input/output buffer allocation)
        /// \param keep_ratio Whether to keep original aspect ratio. This is good for accuracy, but requires extra steps to refine the `hyperpose::human_t`.
        /// \param factor For each element in the input data, they will be multiplied by "factor".
        /// \param flip_rgb Whether to convert the color channels from "BGR" to "RGB".
        explicit tensorrt(const tensorrt_serialized& serialized_model, cv::Size input_size, int max_batch_size = 8,
            bool keep_ratio = false,
            double factor = 1. / 255, bool flip_rgb = true);

        /// Deconstructor of class hyperpose::dnn::tensorrt.
        ~tensorrt();

        ///
        /// \return The maximum batch size of this engine.
        inline int max_batch_size() noexcept { return m_max_batch_size; }

        ///
        /// \return The input `(width, height)` of this engine.
        inline cv::Size input_size() noexcept { return m_inp_size; }

        /// Do inference with `cv::Mat`(OpenCV image/matrix data structure).
        /**
         * @code
         * namespace pp = hyperpose;
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
        /// \return A vector of output feature maps(tensors), ordered by tensor name.
        std::vector<internal_t> inference(std::vector<cv::Mat> inputs);

        /// \brief Do inference using plain float buffers(NCHW format required).
        /// \details This step will not involve in any scalar multiplication or channel swapping(related to the `factor`
        /// and `flip_rgb` parameter in the constructor).
        ///
        /// \see hyperpose::nhwc_images_append_nchw_batch
        /// \see hyperpose::tensorrt::tensorrt
        ///
        /// \param float_buffer The input float buffers.
        /// \param batch_size The batch size of inputs to do inference.
        /// \return  vector of output feature maps(tensors), ordered by tensor name.
        std::vector<internal_t> inference(const std::vector<float>& float_buffer, size_t batch_size);

        /// Save the TensorRT engine to serialized protobuf format.
        /// \param path Path to serialized engine model file.
        void save(const std::string path);

    private:
        const cv::Size m_inp_size; // w, h
        const int m_max_batch_size;
        const bool m_keep_ratio;
        const double m_factor;
        const bool m_flip_rgb;

        // Cuda related.
        struct cuda_dep;
        std::unique_ptr<cuda_dep> m_cuda_dep;

        bool m_binding_has_batch_dim = true;

    private:
        void _batching(std::vector<cv::Mat>&, std::vector<float>&);
        void _create_binding_buffers();
    };

} // namespace dnn

} // namespace hyperpose