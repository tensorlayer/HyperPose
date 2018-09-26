#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvUffParser.h>
#include <NvUtils.h>

#include "cuda_buffer.h"
#include "logger.h"
#include "tracer.h"
#include "uff-runner.h"

constexpr uint64_t Gi = 1 << 30;

using input_info_t = std::vector<std::pair<std::string, std::vector<int>>>;

Logger gLogger;

inline int64_t volume(const nvinfer1::Dims &d)
{
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; i++) v *= d.d[i];
    return v;
}

std::string to_string(const nvinfer1::Dims &d)
{
    std::string s;
    for (int64_t i = 0; i < d.nbDims; i++) {
        if (!s.empty()) { s += ", "; }
        s += std::to_string(d.d[i]);
    }
    return "(" + s + ")";
}

std::string to_string(const nvinfer1::DataType dtype)
{
    return std::to_string(int(dtype));
}

template <typename T> struct destroy_deleter {
    void operator()(T *ptr) { ptr->destroy(); }
};

template <typename T>
using destroy_ptr = std::unique_ptr<T, destroy_deleter<T>>;

nvinfer1::ICudaEngine *loadModelAndCreateEngine(const char *uffFile,
                                                int maxBatchSize,
                                                nvuffparser::IUffParser *parser,
                                                bool use_f16 = false)
{
    destroy_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(gLogger));
    destroy_ptr<nvinfer1::INetworkDefinition> network(builder->createNetwork());

    if (use_f16) {
        if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF)) {
            // RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
            return nullptr;
        }
        builder->setFp16Mode(true);
    } else {
        if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT)) {
            // RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
            return nullptr;
        }
    }

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 * Gi);

    nvinfer1::ICudaEngine *engine = builder->buildCudaEngine(*network);
    if (!engine) {
        return nullptr;
        // RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    }
    return engine;
}

nvinfer1::ICudaEngine *
create_engine(const std::string &model_file, const input_info_t &input_info,
              const std::vector<std::string> &output_names, int maxBatchSize)
{
    TRACE(__func__);
    destroy_ptr<nvuffparser::IUffParser> parser(nvuffparser::createUffParser());
    for (const auto &info : input_info) {
        const auto dims = info.second;
        parser->registerInput(
            info.first.c_str(),
            // Always provide your dimensions in CHW even if your
            // network input was in HWC in yout original framework.
            nvinfer1::DimsCHW(dims[0], dims[1], dims[2]),
            nvuffparser::UffInputOrder::kNCHW  //
        );
    }
    for (auto &name : output_names) { parser->registerOutput(name.c_str()); }
    nvinfer1::ICudaEngine *engine = loadModelAndCreateEngine(
        model_file.c_str(), maxBatchSize, parser.get());
    if (!engine) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                    "failed to created engine");
        exit(1);
    }
    return engine;
}

class UFFRunnerImpl : public UFFRunner
{
  public:
    UFFRunnerImpl(const std::string &model_file, const input_info_t &input_info,
                  const std::vector<std::string> &output_names,
                  int maxBatchSize);
    ~UFFRunnerImpl() override;

    void execute(const std::vector<void *> &inputs,
                 const std::vector<void *> &outputs, int batchSize) override;

  private:
    destroy_ptr<nvinfer1::ICudaEngine> engine_;
    std::vector<std::unique_ptr<cuda_buffer_t>> buffers_;

    void createBuffers_(int batchSize);
};

UFFRunnerImpl::UFFRunnerImpl(const std::string &model_file,
                             const input_info_t &input_info,
                             const std::vector<std::string> &output_names,
                             int maxBatchSize)
    : engine_(create_engine(model_file, input_info, output_names, maxBatchSize))
{
    createBuffers_(maxBatchSize);
}

UFFRunnerImpl::~UFFRunnerImpl() { nvuffparser::shutdownProtobufLibrary(); }

void UFFRunnerImpl::createBuffers_(int batchSize)
{
    TRACE(__func__);

    const int n = engine_->getNbBindings();
    for (int i = 0; i < n; ++i) {
        const nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        const nvinfer1::DataType dtype = engine_->getBindingDataType(i);
        const std::string name(engine_->getBindingName(i));

        std::cout << "binding " << i << ":"
                  << " name: " << name << " type" << to_string(dtype)
                  << to_string(dims) << std::endl;
        const auto info = buffer_info_t{volume(dims) * batchSize, dtype};
        buffers_.push_back(
            std::unique_ptr<cuda_buffer_t>(new cuda_buffer_t(info)));
    }
}

void UFFRunnerImpl::execute(const std::vector<void *> &inputs,
                            const std::vector<void *> &outputs, int batchSize)
{
    TRACE(__func__);

    {
        TRACE("copy input from host");
        int idx = 0;
        for (int i = 0; i < buffers_.size(); ++i) {
            if (engine_->bindingIsInput(i)) {
                buffers_[i]->fromHost(inputs[idx++]);
            }
        }
    }

    {
        TRACE("context->execute");
        auto context = engine_->createExecutionContext();
        std::vector<void *> buffer_ptrs_(buffers_.size());
        for (int i = 0; i < buffers_.size(); ++i) {
            buffer_ptrs_[i] = buffers_[i]->data();
        }
        context->execute(batchSize, buffer_ptrs_.data());
        context->destroy();
    }

    {
        TRACE("copy output to host");
        int idx = 0;
        for (int i = 0; i < buffers_.size(); ++i) {
            if (!engine_->bindingIsInput(i)) {
                buffers_[i]->toHost(outputs[idx++]);
            }
        }
    }
}

UFFRunner *create_openpose_runner(const std::string &model_file,
                                  int input_height, int input_width,
                                  int maxBatchSize)
{
    const input_info_t input_info = {
        {
            "image",
            {3, input_height, input_width} /* must be (C, H, W) */,
        },
    };

    const std::vector<std::string> output_names = {
        "outputs/conf",
        "outputs/paf",
    };
    return new UFFRunnerImpl(model_file, input_info, output_names,
                             maxBatchSize);
}
