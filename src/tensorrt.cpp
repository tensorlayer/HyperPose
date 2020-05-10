#include <openpose_plus/operator/dnn/tensorrt.hpp>
#include <openpose_plus/utility/data.hpp>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvUffParser.h>

#include <ttl/experimental/copy>
#include <ttl/range>

#include "logger.h"
#include "logging.hpp"
#include "trace.hpp"

namespace ttl {
template <typename R, rank_t r>
tensor_ref<char, 1> ref_chars(const ttl::tensor<R, r>& t)
{
    // TODO: make it an offical API: https://github.com/stdml/stdtensor/issues/61
    return tensor_ref<char, 1>(reinterpret_cast<char*>(t.data()), t.data_size());
}
}

namespace poseplus {
namespace dnn {

    template <typename T>
    struct engine_deleter {
        void operator()(T* ptr) { ptr->destroy(); }
    };

    template <typename T>
    using destroy_ptr = std::unique_ptr<T, engine_deleter<T>>;

    Logger gLogger;

    inline int64_t volume(const nvinfer1::Dims& d)
    {
        int64_t v = 1;
        for (int i = 0; i < d.nbDims; i++) {
            v *= d.d[i];
        }
        return v;
    }

    inline size_t sizeof_element(nvinfer1::DataType t)
    {
        size_t ret = 0;
        switch (t) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            ret = 4;
            break;
        case nvinfer1::DataType::kHALF:
            ret = 2;
            break;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kBOOL: // TODO. Validation.
            ret = 1;
            break;
        }
        assert(ret != 0);
        return ret;
    }

    std::string to_string(const nvinfer1::Dims& d)
    {
        std::string s{ "(" };
        if (d.nbDims != 0) {
            for (int64_t i = 0; i < d.nbDims; i++)
                (s += std::to_string(d.d[i])) += ", ";
            s.pop_back();
            s.pop_back();
        }
        return s + ')';
    }

    std::string to_string(const nvinfer1::DataType dtype)
    {
        std::string ret;
        switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            ret = "float32";
            break;
        case nvinfer1::DataType::kINT32:
            ret = "int32";
            break;
        case nvinfer1::DataType::kHALF:
            ret = "float16";
            break;
        case nvinfer1::DataType::kINT8:
            ret = "int8";
            break;
        case nvinfer1::DataType::kBOOL: // TODO. Validation.
            ret = "bool";
            break;
        }
        return ret;
    }

    // * Create TensorRT engine.
    static nvinfer1::ICudaEngine*
    create_uff_engine(const std::string& model_file, cv::Size input_size,
        const std::string& input_name,
        const std::vector<std::string>& output_names, int max_batch_size,
        nvinfer1::DataType dtype)
    {
        TRACE_SCOPE(__func__);
        destroy_ptr<nvuffparser::IUffParser> parser(nvuffparser::createUffParser());

        parser->registerInput(
            input_name.c_str(),
            nvinfer1::DimsCHW(3, input_size.height, input_size.width),
            nvuffparser::UffInputOrder::kNCHW);

        for (auto& name : output_names) {
            parser->registerOutput(name.c_str());
        }

        destroy_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
        destroy_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
        if (!parser->parse(model_file.c_str(), *network, dtype)) {
            gLogger.log(
                nvinfer1::ILogger::Severity::kERROR,
                ("Failed to parse Uff in data type: " + to_string(dtype)).c_str());
            exit(1);
        }

        builder->setMaxBatchSize(max_batch_size);

        destroy_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        config->setMaxWorkspaceSize((1 << 20) * 512);
        auto engine = builder->buildEngineWithConfig(*network, *config);

        if (nullptr == engine) {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                "Failed to created engine");
            exit(1);
        }
        return engine;
    }

    static nvinfer1::ICudaEngine*
    create_onnx_engine(const std::string& model_file, int max_batch_size, nvinfer1::DataType dtype)
    {
        TRACE_SCOPE(__func__);
        destroy_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
        const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        destroy_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
        destroy_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));

        if (!parser->parseFromFile(model_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            gLogger.log(
                nvinfer1::ILogger::Severity::kERROR,
                ("Failed to parse ONNX in data type: " + to_string(dtype)).c_str());
            exit(1);
        }

        builder->setMaxBatchSize(max_batch_size);
        destroy_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        config->setMaxWorkspaceSize((1 << 20) * 512); // TODO: A better way to set the workspace.

        auto engine = builder->buildEngineWithConfig(*network, *config);

        if (nullptr == engine) {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                "Failed to created engine");
            exit(1);
        }
        return engine;
    }

    // * Class impl.
    tensorrt::tensorrt(const uff& uff_model, cv::Size input_size,
        int max_batch_size, nvinfer1::DataType dtype, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_factor(factor)
        , m_engine(create_uff_engine(uff_model.model_path, input_size, uff_model.input_name, uff_model.output_names,
              max_batch_size, dtype))
    {
        // Creat buffers.
        const auto n_bind = m_engine->getNbBindings();
        m_cuda_buffers.reserve(n_bind);
        for (auto i : ttl::range(n_bind)) {
            const nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
            const nvinfer1::DataType data_type = m_engine->getBindingDataType(i);
            const std::string name(m_engine->getBindingName(i));
            info("Binding ", i, ':', " name: ", name, " @type ", to_string(data_type), " @shape ", to_string(dims), '\n');
            m_cuda_buffers.emplace_back(max_batch_size,
                volume(dims) * sizeof_element(data_type));
        }
    }

    tensorrt::tensorrt(const onnx& onnx_model, cv::Size input_size,
        int max_batch_size, nvinfer1::DataType dtype, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_factor(factor)
        , m_engine(create_onnx_engine(onnx_model.model_path, max_batch_size, dtype))
    {
        // Creat buffers.
        const auto n_bind = m_engine->getNbBindings();
        m_cuda_buffers.reserve(n_bind);
        for (auto i : ttl::range(n_bind)) {
            const nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
            const nvinfer1::DataType data_type = m_engine->getBindingDataType(i);
            const std::string name(m_engine->getBindingName(i));
            info("Binding ", i, ':', " name: ", name, " @type ", to_string(data_type), " @shape ", to_string(dims), '\n');
            m_cuda_buffers.emplace_back(max_batch_size, volume(dims) * sizeof_element(data_type));
        }
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
        TRACE_SCOPE("INFERENCE::TensorRT");
        {
            TRACE_SCOPE("INFERENCE::TensorRT::host2dev");
            for (auto i : ttl::range(m_cuda_buffers.size()))
                if (m_engine->bindingIsInput(i)) {
                    info("Got Input Binding! ", i, '\n');
                    const auto buffer = m_cuda_buffers.at(i).slice(0, batch_size);
                    ttl::tensor_view<char, 2> input(
                        reinterpret_cast<const char*>(cpu_image_batch_buffer.data()),
                        buffer.shape());
                    ttl::copy(buffer, input);
                    break; // ! Only one input node.
                }
        }

        {
            TRACE_SCOPE("INFERENCE::TensorRT::context->execute");
            auto context = m_engine->createExecutionContext();
            std::vector<void*> buffer_ptrs_(m_cuda_buffers.size());
            std::transform(m_cuda_buffers.begin(), m_cuda_buffers.end(),
                buffer_ptrs_.begin(),
                [](const auto& b) { return b.data(); });
            context->execute(batch_size, buffer_ptrs_.data());
            context->destroy();
        }

        {
            TRACE_SCOPE("INFERENCE::TensorRT::dev2host");
            for (auto i : ttl::range(m_cuda_buffers.size())) {
                if (!m_engine->bindingIsInput(i)) {
                    const auto buffer = m_cuda_buffers[i].slice(0, batch_size);

                    const nvinfer1::Dims out_dims = m_engine->getBindingDimensions(i);
                    const ttl::shape<3> feature_shape(out_dims.d[0], out_dims.d[1], out_dims.d[2]);

                    auto name = m_engine->getBindingName(i);

                    info("Get Inference Result: ", name, ": ", to_string(out_dims), '\n');

                    for (auto j : ttl::range(batch_size)) {
                        ttl::tensor<float, 3> host_tensor(feature_shape);
                        ttl::copy(ttl::ref_chars(host_tensor), ttl::view(buffer[j]));
                        ret[j].emplace_back(name, std::move(host_tensor));
                    }
                }
            }
        }
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

    tensorrt::~tensorrt() { nvuffparser::shutdownProtobufLibrary(); }

} // namespace dnn

} // namespace poseplus
