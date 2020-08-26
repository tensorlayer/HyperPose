#include <hyperpose/operator/dnn/tensorrt.hpp>
#include <hyperpose/utility/data.hpp>

#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <ttl/cuda_tensor>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvUffParser.h>

#include <ttl/experimental/copy>
#include <ttl/range>

#include "logger.h"
#include "logging.hpp"
#include "trace.hpp"
#include <algorithm>

namespace ttl {
template <typename R, rank_t r>
tensor_ref<char, 1> ref_chars(const ttl::tensor<R, r>& t)
{
    // TODO: make it an offical API: https://github.com/stdml/stdtensor/issues/61
    return tensor_ref<char, 1>(reinterpret_cast<char*>(t.data()), t.data_size());
}
}

namespace hyperpose {
namespace dnn {

    template <typename T>
    struct engine_deleter {
        void operator()(T* ptr) { ptr->destroy(); }
    };

    template <typename T>
    using destroy_ptr = std::unique_ptr<T, engine_deleter<T>>;

    Logger gLogger;

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

    inline int64_t volume(const nvinfer1::Dims& d)
    {
        int64_t v = 1;
        for (int i = 0; i < d.nbDims; i++) {
            v *= d.d[i];
        }
        return v;
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

    struct tensorrt::cuda_dep {
        using cuda_buffer_t = ttl::cuda_tensor<char, 2>; // [batch_size, data_size]

        std::unordered_map<std::string, cuda_buffer_t> m_cuda_buffers;
        destroy_ptr<nvinfer1::ICudaEngine> m_engine;
        destroy_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

        explicit cuda_dep(nvinfer1::ICudaEngine* ptr)
            : m_engine(ptr)
            , m_context(m_engine->createExecutionContext())
        {
        }
    };

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
            nvinfer1::Dims3(3, input_size.height, input_size.width),
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
        config->setMaxWorkspaceSize(1ull << 30);
        auto engine = builder->buildEngineWithConfig(*network, *config);

        if (nullptr == engine) {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                "Failed to created engine");
            exit(1);
        }
        return engine;
    }

    static nvinfer1::ICudaEngine*
    create_onnx_engine(const std::string& model_file, int max_batch_size, nvinfer1::DataType dtype, cv::Size size)
    {
        TRACE_SCOPE(__func__);
        destroy_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));

        const auto build_flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        destroy_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(build_flag));
        destroy_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));

        if (!parser->parseFromFile(model_file.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            gLogger.log(
                nvinfer1::ILogger::Severity::kERROR,
                ("Failed to parse ONNX model in data type: " + to_string(dtype)).c_str());
            exit(-1);
        }

        if (network->getNbInputs() != 1)
            error("Detected multiple inputs. (Only support one-input mode currently)\n");

        destroy_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        config->setMaxWorkspaceSize(1ull << 20); // TODO: A better way to set the workspace.

        builder->setMaxBatchSize(max_batch_size);

        auto dims = network->getInput(0)->getDimensions();

        if (dims.nbDims < 3 || dims.nbDims > 4) {
            error("Dimension error: Expected: ", to_string(dims), ", accepted: ", to_string(nvinfer1::Dims4(-1, 3, size.height, size.width)));
        }

        if ((dims.nbDims == 3 && dims.d[0] != 3) || (dims.nbDims == 4 && dims.d[1] != 3)) {
            error("Dimension error(Channel dimension must be 3): Expected: ", to_string(dims), ", accepted: ", to_string(nvinfer1::Dims4(-1, 3, size.height, size.width)));
        }

        info("Network Input Dimensions: Name@", network->getInput(0)->getName(), ", Dims@", to_string(network->getInput(0)->getDimensions()), '\n');

        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        profile->setDimensions(
            network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, size.height, size.width));
        profile->setDimensions(
            network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, 3, size.height, size.width));
        profile->setDimensions(
            network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, 3, size.height, size.width));

        config->addOptimizationProfile(profile);

        info("Started profiling and engine building.\n");
        auto engine = builder->buildEngineWithConfig(*network, *config);

        info("Profile Info: Minimum input shape: ", to_string(engine->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMIN)), '\n');
        info("Profile Info: Optimum input shape: ", to_string(engine->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kOPT)), '\n');
        info("Profile Info: Maximum input shape: ", to_string(engine->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX)), '\n');

        if (nullptr == engine) {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to created engine");
            exit(1);
        }

        info("Succeed in engine building.\n");
        return engine;
    }

    static nvinfer1::ICudaEngine* create_serialized_engine(const std::string& model_file)
    {
        std::ifstream in_file(model_file, std::ios::binary | std::ios::in);

        std::streampos begin, end;
        begin = in_file.tellg();
        in_file.seekg(0, std::ios::end);
        end = in_file.tellg();

        const std::size_t size = end - begin;
        info("engine file size: ", size, " bytes\n");

        in_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_data(new char[size]);
        in_file.read((char*)engine_data.get(), size);
        in_file.close();

        // deserialize the engine
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine((const void*)engine_data.get(), size, nullptr);

        if (nullptr == engine) {
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to created engine");
            exit(1);
        }

        return engine;
    }

    // * Class impl.
    void tensorrt::_create_binding_buffers()
    {
        // Creat buffers.
        const auto n_bind = m_cuda_dep->m_engine->getNbBindings();
        m_cuda_dep->m_cuda_buffers.reserve(n_bind);
        for (auto i : ttl::range(n_bind))
            if (m_cuda_dep->m_engine->bindingIsInput(i)) {
                const nvinfer1::Dims dims = m_cuda_dep->m_engine->getBindingDimensions(i);
                const std::array<int, 3> size_vector = { 3, m_inp_size.height, m_inp_size.width };

                auto nn_dims_string = to_string(dims);
                auto input_dims_string = '[' + std::to_string(size_vector[0]) + ", " + std::to_string(size_vector[1]) + ", " + std::to_string(size_vector[2]) + ']';

                if (dims.nbDims < 3 || dims.nbDims > 4)
                    error("The input/output dimension size only allows 3(CHW) or 4(NCHW)\n");

                if (dims.nbDims == 3)
                    m_binding_has_batch_dim = false;

                std::vector<int> compare_vec;
                compare_vec.reserve(dims.nbDims);
                for (auto j : ttl::range(dims.nbDims))
                    compare_vec.push_back(dims.d[j]);

                // Binding CHW must be equal to [3, h, w]
                if (!std::equal(size_vector.rbegin(), size_vector.rend(), compare_vec.rbegin()))
                    error("Input shape mismatch: Network Input Shape: ", nn_dims_string, ", Input shape: ", input_dims_string, '\n');

                const auto data_type = m_cuda_dep->m_engine->getBindingDataType(i);
                const std::string name(m_cuda_dep->m_engine->getBindingName(i));

                info("Binding from TensorRT: Name@", name, ", Type@", to_string(data_type), ", Shape@", nn_dims_string, '\n');
                info("Preallocate memory shape:(CHW) = ", input_dims_string, '\n');

                m_cuda_dep->m_cuda_buffers.emplace(name, cuda_dep::cuda_buffer_t(m_max_batch_size, std::accumulate(size_vector.begin(), size_vector.end(), 1, std::multiplies<int>{}) * sizeof_element(data_type)));
            }

        for (auto i : ttl::range(m_cuda_dep->m_engine->getNbBindings()))
            if (m_cuda_dep->m_engine->bindingIsInput(i)) {
                constexpr int batch_one = 1; // Will be used to calculate the volume.
                if (m_binding_has_batch_dim)
                    m_cuda_dep->m_context->setBindingDimensions(0, nvinfer1::Dims4(batch_one, 3, m_inp_size.height, m_inp_size.width));
                break;
            }

        { // Hook for output shape.
            for (auto i : ttl::range(m_cuda_dep->m_engine->getNbBindings()))
                if (!m_cuda_dep->m_engine->bindingIsInput(i)) {
                    auto dims = m_cuda_dep->m_context->getBindingDimensions(i);

                    auto name = m_cuda_dep->m_engine->getBindingName(i);
                    auto data_type = m_cuda_dep->m_engine->getBindingDataType(i);

                    auto batch_slice_alloc_size = volume(dims) * sizeof_element(data_type);

                    info("Binding from TensorRT: Name@", name, ", Type@", to_string(data_type), ", Shape@", to_string(dims), '\n');
                    info("Preallocate memory size:(Batch x BatchSliceSize) = [", m_max_batch_size, 'x', batch_slice_alloc_size, "] bytes\n");

                    m_cuda_dep->m_cuda_buffers.emplace(name, cuda_dep::cuda_buffer_t(m_max_batch_size, batch_slice_alloc_size));
                }
        }
    }

    tensorrt::tensorrt(const uff& uff_model, cv::Size input_size,
        int max_batch_size, bool keep_ratio, data_type dtype, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_keep_ratio(keep_ratio)
        , m_factor(factor)
        , m_cuda_dep(std::make_unique<cuda_dep>(create_uff_engine(uff_model.model_path, input_size, uff_model.input_name, uff_model.output_names,
              max_batch_size, static_cast<nvinfer1::DataType>(dtype.val))))
    {
        _create_binding_buffers();
    }

    tensorrt::tensorrt(const tensorrt_serialized& serialized_model, cv::Size input_size,
        int max_batch_size, bool keep_ratio, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_keep_ratio(keep_ratio)
        , m_factor(factor)
        , m_cuda_dep(std::make_unique<cuda_dep>(create_serialized_engine(serialized_model.model_path)))
    {
        _create_binding_buffers();
    }

    tensorrt::tensorrt(const onnx& onnx_model, cv::Size input_size,
        int max_batch_size, bool keep_ratio, data_type dtype, double factor,
        bool flip_rgb)
        : m_inp_size(input_size)
        , m_flip_rgb(flip_rgb)
        , m_max_batch_size(max_batch_size)
        , m_keep_ratio(keep_ratio)
        , m_factor(factor)
        , m_cuda_dep(std::make_unique<cuda_dep>(create_onnx_engine(onnx_model.model_path, max_batch_size, static_cast<nvinfer1::DataType>(dtype.val), input_size)))
    {
        _create_binding_buffers();
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
            for (auto i : ttl::range(m_cuda_dep->m_engine->getNbBindings()))
                if (m_cuda_dep->m_engine->bindingIsInput(i)) {
                    auto name = m_cuda_dep->m_engine->getBindingName(i);
                    const auto buffer = m_cuda_dep->m_cuda_buffers.at(name).slice(0, batch_size);

                    if (m_binding_has_batch_dim)
                        m_cuda_dep->m_context->setBindingDimensions(0, nvinfer1::Dims4(batch_size, 3, m_inp_size.height, m_inp_size.width));

                    info("Got Input Binding! ", 0, '\n');
                    ttl::tensor_view<char, 2> input(
                        reinterpret_cast<const char*>(cpu_image_batch_buffer.data()),
                        buffer.shape());
                    ttl::copy(buffer, input);
                }
        }

        {
            TRACE_SCOPE("INFERENCE::TensorRT::context->execute");
            std::vector<void*> buffer_ptrs;
            for (auto i : ttl::range(m_cuda_dep->m_engine->getNbBindings()))
                buffer_ptrs.push_back(m_cuda_dep->m_cuda_buffers.at(m_cuda_dep->m_engine->getBindingName(i)).data());
            if (m_binding_has_batch_dim)
                m_cuda_dep->m_context->executeV2(buffer_ptrs.data());
            else
                m_cuda_dep->m_context->execute(m_max_batch_size, buffer_ptrs.data());
        }

        {
            TRACE_SCOPE("INFERENCE::TensorRT::dev2host");

            std::vector<std::pair<int, std::string>> output_names;
            for (auto i : ttl::range(m_cuda_dep->m_cuda_buffers.size()))
                if (!m_cuda_dep->m_engine->bindingIsInput(i))
                    output_names.emplace_back(i, m_cuda_dep->m_engine->getBindingName(i));
            std::sort(output_names.begin(), output_names.end(), [](auto& l, auto& r) { return l.second < r.second; });

            for (auto&& p : output_names) {
                int i = p.first;
                auto& name = p.second;

                const auto buffer = m_cuda_dep->m_cuda_buffers.at(name).slice(0, batch_size);

                const nvinfer1::Dims out_dims = m_cuda_dep->m_engine->getBindingDimensions(i);

                const size_t start_index = m_binding_has_batch_dim ? 1 : 0;
                std::vector<int> non_batch_shape;
                non_batch_shape.reserve(out_dims.nbDims - start_index);
                for (size_t k = start_index; k < out_dims.nbDims; ++k)
                    non_batch_shape.push_back(out_dims.d[k]);

                info("Get Inference Result: ", name, ": ", to_string(out_dims), '\n');

                for (auto j : ttl::range(batch_size)) {
                    auto [slice_size] = buffer[j].dims();
                    std::unique_ptr<char[]> data{ new char[slice_size] };

                    ttl::copy(ttl::vector_ref<char>(data.get(), ttl::shape<1>(slice_size)), ttl::view(buffer[j]));
                    ret[j].emplace_back(name, std::move(data), non_batch_shape);
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
        for (auto&& mat : batch) {
            if (m_keep_ratio)
                mat = non_scaling_resize(mat, m_inp_size); // This involves in copy.
            else
                cv::resize(mat, mat, m_inp_size);
        }

        thread_local std::vector<float> cpu_image_batch_buffer;
        cpu_image_batch_buffer.clear();

        // * Step2: NHWC -> NCHW && Batching,
        this->_batching(batch, cpu_image_batch_buffer);

        // * Step3: Do Inference.
        return this->inference(cpu_image_batch_buffer, batch.size());
    }

    void tensorrt::save(const std::string path)
    {
        destroy_ptr<nvinfer1::IHostMemory> serializedModel(m_cuda_dep->m_engine->serialize());
        std::ofstream ofs(path, std::ios::out | std::ios::binary);
        info("Writing ", serializedModel->size(), " bytes to model path: ", path, '\n');
        ofs.write(static_cast<char*>(serializedModel->data()), serializedModel->size());
        info("Serialized engine built successfully!\n");
        ofs.close();
    }

    tensorrt::~tensorrt() = default;

} // namespace dnn

} // namespace hyperpose
