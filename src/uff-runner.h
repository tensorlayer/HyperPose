#pragma once

#include <string>
#include <vector>

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
