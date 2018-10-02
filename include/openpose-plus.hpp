// The public C++ API of openpose-plus
#pragma once
#include <openpose-plus/human.h>

class uff_runner
{
  public:
    virtual void execute(const std::vector<void *> &inputs,
                         const std::vector<void *> &outputs,
                         int batchSize = 1) = 0;
    virtual ~uff_runner() {}
};

uff_runner *create_openpose_runner(const std::string &model_file,
                                   int input_height, int input_width,
                                   int max_batch_size, bool use_f16);

class paf_processor
{
  public:
    virtual std::vector<human_t> operator()(const float * /* heatmap */,
                                            const float * /* PAF */,
                                            bool /* use GPU */) = 0;

    virtual ~paf_processor() {}
};

paf_processor *create_paf_processor(int input_height, int input_width,
                                    int height, int width, int n_joins,
                                    int n_connections, int gauss_kernel_size);
