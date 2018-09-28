#include <chrono>
#include <thread>

#include "channel.hpp"
#include "input.h"
#include "paf.h"
#include "stream_detector.h"
#include "tensor.h"
#include "tracer.h"
#include "uff-runner.h"

struct stop_watch_t {
    using clock_t = std::chrono::system_clock;
    using duration_t = std::chrono::duration<double>;

    std::chrono::time_point<clock_t> t0;

    stop_watch_t() : t0(clock_t::now()) {}

    duration_t tick()
    {
        const auto t1 = clock_t::now();
        const duration_t d = t1 - t0;
        t0 = t1;
        return d;
    }
};

class stream_detector_impl : public stream_detector
{
  public:
    stream_detector_impl(const std::string &model_file,          //
                         int input_height, int input_width,      //
                         int feature_height, int feature_width,  //
                         int buffer_size, bool use_f16, int gauss_kernel_size)
        : buffer_size(buffer_size),
          image_stream_in(buffer_size),
          image_stream_out(buffer_size),
          feature_stream_in(buffer_size),
          feature_stream_out(buffer_size),
          height(input_height),
          width(input_width),
          feature_height(feature_height),
          feature_width(feature_width),
          images(nullptr, buffer_size, 3, height, width),
          confs(nullptr, buffer_size, n_joins, feature_height, feature_width),
          pafs(nullptr, buffer_size, n_connections * 2, feature_height,
               feature_width),
          paf_process(create_paf_processor(feature_height, feature_width,
                                           input_height, input_width, n_joins,
                                           n_connections, gauss_kernel_size)),
          runner(create_openpose_runner(model_file, height, width, 1, use_f16))
    {
    }

    void run(const std::vector<std::string> &filenames) override
    {
        TRACE(__func__);
        std::vector<std::thread> ths;

        for (int i = 0; i < buffer_size; ++i) {
            image_stream_in.put(images.kth_slice(i));
            feature_stream_in.put(
                std::make_pair(confs.kth_slice(i), pafs.kth_slice(i)));
        }

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < filenames.size(); ++i) {
                const auto ptr = image_stream_in.get();
                input_image(filenames[i], height, width, ptr);
                image_stream_out.put(ptr);
            }
        }));

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < filenames.size(); ++i) {
                const auto ptr = image_stream_out.get();
                const auto pr = feature_stream_in.get();
                runner->execute({ptr}, {pr.first, pr.second}, 1);
                image_stream_in.put(ptr);
                feature_stream_out.put(pr);
            }
        }));

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < filenames.size(); ++i) {
                const auto pr = feature_stream_out.get();
                const auto humans = (*paf_process)(pr.first, pr.second, false);
                feature_stream_in.put(pr);
                printf("got %lu humnas from %d-th image\n", humans.size(), i);
            }
        }));

        for (auto &th : ths) { th.join(); }
    }

  private:
    const int buffer_size;

    channel<float *> image_stream_in;
    channel<float *> image_stream_out;

    channel<std::pair<float *, float *>> feature_stream_in;
    channel<std::pair<float *, float *>> feature_stream_out;

    const int height;
    const int width;

    const int feature_height;
    const int feature_width;

    tensor_t<float, 4> images;
    tensor_t<float, 4> confs;
    tensor_t<float, 4> pafs;

    std::unique_ptr<paf_processor> paf_process;
    std::unique_ptr<uff_runner> runner;
};

stream_detector *stream_detector::create(const std::string &model_file,
                                         int input_height, int input_width,  //
                                         int feature_height, int feature_width,
                                         int buffer_size, bool use_f16,
                                         int gauss_kernel_size)
{
    return new stream_detector_impl(model_file, input_height, input_width,
                                    feature_height, feature_width, buffer_size,
                                    use_f16, gauss_kernel_size);
}
