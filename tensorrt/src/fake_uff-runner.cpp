#include "tracer.h"
#include "uff-runner.h"

class FakeUFFRunnerImpl : public UFFRunner
{
  public:
    FakeUFFRunnerImpl(const std::string &model_file) { tracer_t _(__func__); }

    void execute(const std::vector<void *> &inputs,
                 const std::vector<void *> &outputs) override
    {
        tracer_t _(__func__);
    }
};

UFFRunner *create_runner(const std::string &model_file)
{
    return new FakeUFFRunnerImpl(model_file);
}
