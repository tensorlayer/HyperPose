#include <algorithm>
#include <cassert>
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

#include <openpose-plus.h>

#include "logger.h"
#include "std_cuda_tensor.hpp"
#include "trace.hpp"

using input_info_t = std::vector<std::pair<std::string, std::vector<int>>>;

Logger gLogger;

inline int64_t volume(const nvinfer1::Dims &d)
{
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; i++) v *= d.d[i];
    return v;
}

inline size_t elementSize(nvinfer1::DataType t)
{
    switch (t) {
    // TODO: check nvinfer1 version
    // case nvinfer1::DataType::kINT32:
    //     return 4;
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT8:
        return 1;
    }
    assert(0);
    return 0;
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
                                                int max_batch_size,
                                                nvuffparser::IUffParser *parser,
                                                bool use_f16)
{
    destroy_ptr<nvinfer1::IBuilder> builder(
        nvinfer1::createInferBuilder(gLogger));
    destroy_ptr<nvinfer1::INetworkDefinition> network(builder->createNetwork());

    if (use_f16) {
        if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF)) {
            return nullptr;
        }
        // builder->setFp16Mode(true);
        builder->setHalf2Mode(true);  // For older version of tensorRT
    } else {
        if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT)) {
            return nullptr;
        }
    }
    builder->setMaxBatchSize(max_batch_size);
    return builder->buildCudaEngine(*network);
}

nvinfer1::ICudaEngine *
create_engine(const std::string &model_file, const input_info_t &input_info,
              const std::vector<std::string> &output_names, int max_batch_size,
              bool use_f16)
{
    TRACE_SCOPE(__func__);
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
    auto engine = loadModelAndCreateEngine(model_file.c_str(), max_batch_size,
                                           parser.get(), use_f16);
    if (!engine) {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR,
                    "failed to created engine");
        exit(1);
    }
    return engine;
}

class uff_runner_impl : public pose_detection_runner
{
  public:
    uff_runner_impl(const std::string &model_file,
                    const input_info_t &input_info,
                    const std::vector<std::string> &output_names,
                    int max_batch_size, bool use_f16);
    ~uff_runner_impl() override;

    void operator()(const std::vector<void *> &inputs,
                    const std::vector<void *> &outputs,
                    int batch_size) override;

  private:
    const int max_batch_size;

    destroy_ptr<nvinfer1::ICudaEngine> engine_;

    using cuda_buffer_t = cuda_tensor<char, 1>;
    std::vector<std::unique_ptr<cuda_buffer_t>> buffers_;

    void createBuffers_(int batch_size);
};

uff_runner_impl::uff_runner_impl(const std::string &model_file,
                                 const input_info_t &input_info,
                                 const std::vector<std::string> &output_names,
                                 int max_batch_size, bool use_f16)
    : max_batch_size(max_batch_size),
      engine_(create_engine(model_file, input_info, output_names,
                            max_batch_size, use_f16))
{
    createBuffers_(max_batch_size);
}

uff_runner_impl::~uff_runner_impl() { nvuffparser::shutdownProtobufLibrary(); }

void uff_runner_impl::createBuffers_(int batch_size)
{
    TRACE_SCOPE(__func__);

    const int n = engine_->getNbBindings();
    for (int i = 0; i < n; ++i) {
        const nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        const nvinfer1::DataType dtype = engine_->getBindingDataType(i);
        const std::string name(engine_->getBindingName(i));

        std::cout << "binding " << i << ":"
                  << " name: " << name << " type" << to_string(dtype)
                  << to_string(dims) << std::endl;
        const size_t mem_size = batch_size * volume(dims) * elementSize(dtype);
        buffers_.push_back(
            std::unique_ptr<cuda_buffer_t>(new cuda_buffer_t(mem_size)));
    }
}

void uff_runner_impl::operator()(const std::vector<void *> &inputs,
                                 const std::vector<void *> &outputs,
                                 int batch_size)
{
    TRACE_SCOPE("uff_runner_impl::operator()");
    assert(batch_size <= max_batch_size);

    {
        TRACE_SCOPE("copy input from host");
        int idx = 0;
        for (int i = 0; i < buffers_.size(); ++i) {
            if (engine_->bindingIsInput(i)) {
                buffers_[i]->partialFromHost(inputs[idx++], batch_size,
                                             max_batch_size);
            }
        }
    }

    {
        TRACE_SCOPE("uff_runner_impl::context->execute");
        auto context = engine_->createExecutionContext();
        std::vector<void *> buffer_ptrs_(buffers_.size());
        for (int i = 0; i < buffers_.size(); ++i) {
            buffer_ptrs_[i] = buffers_[i]->data();
        }
        context->execute(batch_size, buffer_ptrs_.data());
        context->destroy();
    }

    {
        TRACE_SCOPE("copy output to host");
        int idx = 0;
        for (int i = 0; i < buffers_.size(); ++i) {
            if (!engine_->bindingIsInput(i)) {
                buffers_[i]->partialToHost(outputs[idx++], batch_size,
                                           max_batch_size);
            }
        }
    }
}

pose_detection_runner *
create_pose_detection_runner(const std::string &model_file, int input_height,
                             int input_width, int max_batch_size, bool use_f16)
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
    return new uff_runner_impl(model_file, input_info, output_names,
                               max_batch_size, use_f16);
}
