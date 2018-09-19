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
#include "debug.h"
#include "logger.h"
#include "tracer.h"
#include "uff-runner.h"

Logger gLogger;

const uint64_t Gi = 1 << 30;

namespace
{
int height = 368;
int width = 432;
int channel = 3;
}  // namespace

inline int64_t volume(const nvinfer1::Dims &d)
{
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; i++) v *= d.d[i];
    return v;
}

class UFFRunnerImpl : public UFFRunner
{
  public:
    UFFRunnerImpl(const std::string &model_file);
    ~UFFRunnerImpl() override;

    void execute(const std::vector<void *> &inputs,
                 const std::vector<void *> &outputs) override;

  private:
    constexpr static int batchSize = 1;

    nvinfer1::ICudaEngine *engine_;
    std::vector<std::unique_ptr<cuda_buffer_t>> buffers_;

    void createBuffers_(int batchSize);
};

nvinfer1::ICudaEngine *loadModelAndCreateEngine(const char *uffFile,
                                                int maxBatchSize,
                                                nvuffparser::IUffParser *parser)
{
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition *network = builder->createNetwork();

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT)) {
        // RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
        return nullptr;
    }
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF)) {
        // RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
        return nullptr;
    }
    builder->setFp16Mode(true);
#endif

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 * Gi);

    nvinfer1::ICudaEngine *engine = builder->buildCudaEngine(*network);
    if (!engine) {
        return nullptr;
        // RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine");
    }

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}

nvinfer1::ICudaEngine *create_engine(const std::string &model_file)
{
    TRACE(__func__);

    auto parser = nvuffparser::createUffParser();
    parser->registerInput(
        "image",
        // Always provide your dimensions in CHW even if your
        // network input was in HWC in yout original framework.
        nvinfer1::Dims3(channel, height, width),
        nvuffparser::UffInputOrder::kNCHW
        // nvuffparser::UffInputOrder::kNHWC
    );

    const std::vector<std::string> output_names = {
        "outputs/conf",
        "outputs/paf",
    };

    for (auto &name : output_names) { parser->registerOutput(name.c_str()); }

    const int maxBatchSize = 1;
    nvinfer1::ICudaEngine *engine =
        loadModelAndCreateEngine(model_file.c_str(), maxBatchSize, parser);

    if (!engine) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                    "failed to created engine");
        exit(1);
    }
    /* we need to keep the memory created by the parser */
    parser->destroy();
    return engine;
}

UFFRunnerImpl::UFFRunnerImpl(const std::string &model_file)
    : engine_(create_engine(model_file))
{
    createBuffers_(batchSize);
}

UFFRunnerImpl::~UFFRunnerImpl()
{
    engine_->destroy();
    nvuffparser::shutdownProtobufLibrary();
}

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
                            const std::vector<void *> &outputs)
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

UFFRunner *create_runner(const std::string &model_file)
{
    return new UFFRunnerImpl(model_file);
}
