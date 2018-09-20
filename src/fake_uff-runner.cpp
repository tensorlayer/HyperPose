#include "tracer.h"
#include "uff-runner.h"

class FakeUFFRunnerImpl : public UFFRunner
{
  public:
    FakeUFFRunnerImpl(const std::string &model_file) { TRACE(__func__); }

    void execute(const std::vector<void *> &inputs,
                 const std::vector<void *> &outputs, int batchSize) override
    {
        TRACE(__func__);
    }
};

void create_openpose_runner(const std::string &model_file,
                            std::unique_ptr<UFFRunner> &p)
{
    p.reset(new FakeUFFRunnerImpl(model_file));
}
