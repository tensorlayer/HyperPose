#include <swiftpose/operator/dnn/tensorrt.hpp>
#include <swiftpose/utility/data.hpp>

#include <NvInfer.h>
#include <NvUffParser.h>


#include <ttl/experimental/copy>
#include <ttl/range>

#include "logger.h"
#include "trace.hpp"

namespace sp {

namespace dnn{

/*
 * Hided functions.
 */

Logger gLogger;

inline int64_t volume(const nvinfer1::Dims &d)
{
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; i++) { v *= d.d[i]; }
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

std::string to_string(const nvinfer1::Dims &d)
{
    std::string s{"("};
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

// * Initialize the TensorRT engine using the .uff file.
static nvinfer1::ICudaEngine *loadModelAndCreateEngine(const char *uffFile,
                                                       int max_batch_size,
                                                       nvuffparser::IUffParser *parser,
                                                       nvinfer1::DataType dtype)
{
    destroy_ptr<nvinfer1::IBuilder> builder(
            nvinfer1::createInferBuilder(gLogger));
    destroy_ptr<nvinfer1::INetworkDefinition> network(builder->createNetwork());

    if (!parser->parse(uffFile, *network, dtype))
    {
        gLogger.log(
                nvinfer1::ILogger::Severity::kERROR,
                ("Failed to parse Uff in data type: " + to_string(dtype)).c_str());
        return nullptr;
    }

    builder->setMaxBatchSize(max_batch_size);
    return builder->buildCudaEngine(*network);
}

// * Create TensorRT engine.
static nvinfer1::ICudaEngine *
create_engine(
        const std::string &model_file,
        cv::Size input_size, const std::string& input_name,
        const std::vector<std::string>& output_names, int max_batch_size, nvinfer1::DataType dtype)
{
    TRACE_SCOPE(__func__);
    destroy_ptr<nvuffparser::IUffParser> parser(nvuffparser::createUffParser());

    parser->registerInput(
            input_name.c_str(),
            nvinfer1::DimsCHW(3, input_size.height, input_size.width),
            nvuffparser::UffInputOrder::kNCHW);

    for (auto &name : output_names) { parser->registerOutput(name.c_str()); }
    auto engine = loadModelAndCreateEngine(
            model_file.c_str(), max_batch_size, parser.get(), dtype);
    if (nullptr == engine) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to created engine");
        exit(1);
    }
    return engine;
}

// * Class impl.
tensorrt::tensorrt(const std::string &model_path, cv::Size input_size, const std::string& input_name,
                   const std::vector<std::string> output_names,
                   int max_batch_size, nvinfer1::DataType dtype, double factor, bool flip_rgb):
        m_inp_size(input_size),
        m_flip_rgb(flip_rgb),
        m_max_batch_size(max_batch_size),
        m_factor(factor),
        m_engine(create_engine(
                model_path, input_size, input_name, output_names, max_batch_size, dtype))
{
    // Creat buffers.
    const auto n_bind = m_engine->getNbBindings();
    m_cuda_buffers.reserve(n_bind);
    for (auto i : ttl::range(n_bind)) {
        const nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        const nvinfer1::DataType dtype = m_engine->getBindingDataType(i);
        const std::string name(m_engine->getBindingName(i));
        std::cout << "Binding " << i << ':'
                  << " name: " << name << " @type " << to_string(dtype) << " @shape "
                  << to_string(dims) << std::endl;
        m_cuda_buffers.emplace_back(max_batch_size, volume(dims) * sizeof_element(dtype));
    }
}

inference_result_t
tensorrt::sync_inference(const std::vector<cv::Mat>& batch)
{
    TRACE_SCOPE("INFERENCE");
    inference_result_t ret;
    ret.reserve(m_cuda_buffers.size() - 1 /* Input Buffer */);

    if (batch.size() > m_max_batch_size)
    {
        std::string err_msg =
                "Input batch size overflow: Yours@" +
                std::to_string(batch.size()) +
                " Max@" +
                std::to_string(m_max_batch_size);
        gLogger.log(
                nvinfer1::ILogger::Severity::kERROR, err_msg.c_str());
        std::exit(-1);
    }

    // NHWC -> NCHW.
    thread_local std::vector<float> cpu_image_batch_buffer;
    {
        TRACE_SCOPE("INFERENCE::Images2NCHW")
        images2nchw(cpu_image_batch_buffer, batch, m_inp_size, m_factor, m_flip_rgb);
    }

    {
        TRACE_SCOPE("INFERENCE::TensorRT");
        {
            TRACE_SCOPE("INFERENCE::TensorRT::host2dev");
            for (auto i : ttl::range(m_cuda_buffers.size()))
                if (m_engine->bindingIsInput(i)) {
                    const auto buffer = m_cuda_buffers.front().slice(i, batch.size());
                    ttl::tensor_view<char, 2> input(
                            reinterpret_cast<char *>(cpu_image_batch_buffer.data()), buffer.shape());
                    ttl::copy(buffer, input);
                    break; // ! Only one input node.
                }
        }

        {
            TRACE_SCOPE("INFERENCE::TensorRT::context->execute");
            auto context = m_engine->createExecutionContext();
            std::vector<void *> buffer_ptrs_(m_cuda_buffers.size());
            std::transform(m_cuda_buffers.begin(), m_cuda_buffers.end(), buffer_ptrs_.begin(),
                           [](const auto &b) { return b.data(); });
            context->execute(batch.size(), buffer_ptrs_.data());
            context->destroy();
        }

        {
            TRACE_SCOPE("INFERENCE::TensorRT::dev2host");
            int idx = 0;
            for (auto i : ttl::range(m_cuda_buffers.size())) {
                if (!m_engine->bindingIsInput(i)) {
                    const auto buffer = m_cuda_buffers[i].slice(0, batch.size());

                    const nvinfer1::Dims out_dims = m_engine->getBindingDimensions(i);
                    // TODO: Debug
                    std::cout << "Get Inference Result: " << m_engine->getBindingName(i) << ' ' << to_string(out_dims) << std::endl;
                    ret.emplace_back(
                            m_engine->getBindingName(i),
                            ttl::tensor<float, 4>(out_dims.d[0], out_dims.d[1], out_dims.d[2], out_dims.d[3]));

                    ttl::tensor_ref<char, 2> output(
                            reinterpret_cast<char *>(ret.back().tensor().data()), buffer.shape());
                    ttl::copy(output, ttl::view(buffer));
                }
            }
        }

        return ret;
    }
}

tensorrt::~tensorrt() {
    nvuffparser::shutdownProtobufLibrary();
}

}

}