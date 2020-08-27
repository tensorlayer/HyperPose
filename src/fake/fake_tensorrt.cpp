#include <hyperpose/operator/dnn/tensorrt.hpp>
#include <hyperpose/utility/data.hpp>

#include "../logging.hpp"
#include "../trace.hpp"
#include "fake.hpp"
#include <algorithm>

namespace hyperpose {
namespace dnn {

    template <typename T>
    struct engine_deleter {
        void operator()(T* ptr) { ptr->destroy(); }
    };

    template <typename T>
    using destroy_ptr = std::unique_ptr<T, engine_deleter<T>>;

    struct tensorrt::cuda_dep {
    };

    // * Class impl.
    void tensorrt::_create_binding_buffers()
    {
        error_exit_fake();
    }

    tensorrt::tensorrt(const uff& uff_model, cv::Size input_size,
        int max_batch_size, bool keep_ratio, data_type dtype, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_keep_ratio(keep_ratio)
        , m_factor(factor)
    {
        error_exit_fake();
    }

    tensorrt::tensorrt(const tensorrt_serialized& serialized_model, cv::Size input_size,
        int max_batch_size, bool keep_ratio, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_keep_ratio(keep_ratio)
        , m_factor(factor)
    {
        error_exit_fake();
    }

    tensorrt::tensorrt(const onnx& onnx_model, cv::Size input_size,
        int max_batch_size, bool keep_ratio, data_type dtype, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_keep_ratio(keep_ratio)
        , m_factor(factor)
    {
        error_exit_fake();
    }

    void tensorrt::_batching(std::vector<cv::Mat>& batch, std::vector<float>& cpu_image_batch_buffer)
    {
        TRACE_SCOPE("INFERENCE::Images2NCHW");
        nhwc_images_append_nchw_batch(cpu_image_batch_buffer, batch, m_factor, m_flip_rgb);
    }

    std::vector<internal_t>
    tensorrt::inference(const std::vector<float>& cpu_image_batch_buffer, size_t batch_size)
    {
        std::vector<internal_t> ret(batch_size);
        error_exit_fake();
        return ret;
    }

    std::vector<internal_t> tensorrt::inference(std::vector<cv::Mat> batch)
    {
        TRACE_SCOPE("INFERENCE");
        if (batch.size() > m_max_batch_size)
            throw std::logic_error("Input batch size overflow: Yours@"
                + std::to_string(batch.size())
                + " Max@"
                + std::to_string(m_max_batch_size));

        // * Step1: Resize.
        for (auto&& mat : batch)
            cv::resize(mat, mat, m_inp_size); // This involves in copy.

        thread_local std::vector<float> cpu_image_batch_buffer;
        cpu_image_batch_buffer.clear();

        // * Step2: NHWC -> NCHW && Batching,
        this->_batching(batch, cpu_image_batch_buffer);

        // * Step3: Do Inference.
        return this->inference(cpu_image_batch_buffer, batch.size());
    }

    void tensorrt::save(const std::string path)
    {
        error_exit_fake();
    }

    tensorrt::~tensorrt() = default;

} // namespace dnn

} // namespace hyperpose
