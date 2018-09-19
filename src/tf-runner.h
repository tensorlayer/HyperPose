#pragma once

#include <memory>
#include <vector>

class TFRunner
{
  public:
    virtual ~TFRunner() {}

    virtual void operator()(const std::vector<void *> &inputs,
                            const std::vector<void *> &outputs) = 0;
};

void create_openpose_runner(const std::string &model_file, const int height,
                            const int width, std::unique_ptr<TFRunner> &p);
