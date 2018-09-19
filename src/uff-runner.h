#pragma once

#include <memory>
#include <string>
#include <vector>

class UFFRunner
{
  public:
    virtual void execute(const std::vector<void *> &inputs,
                         const std::vector<void *> &outputs,
                         int batchSize = 1) = 0;
    virtual ~UFFRunner() {}
};

UFFRunner *create_runner(const std::string &model_file);
