#include <thread>

#include "stream_detector.h"
#include "uff-runner.h"

struct inputer_t {
    void operator()()
    {
        // TRACE("inputer_t::operator()");
        //
    }
};

struct handler_t {
    void operator()()
    {
        // TRACE("handler_t::operator()");
        //
    }
};

class stream_detector_impl : public stream_detector
{
  public:
    stream_detector_impl() {}

    void run() override
    {
        inputer_t inputer;
        handler_t handler;

        std::thread in(inputer);
        std::thread h(handler);

        in.join();
        h.join();
    }

  private:
};

stream_detector *stream_detector::create(const std::string &model_file,
                                         int input_height,
                                         int input_width,  //
                                         int feature_height, int feature_width,
                                         int batch_size, bool use_f16,
                                         int gauss_kernel_size)
{
    return new stream_detector_impl();
}
