#include <cstdint>
#include <thread>

#include "channel.hpp"
#include "input.h"
#include "paf.h"
#include "stream_detector.h"
#include "tensor.h"
#include "tracer.h"
#include "uff-runner.h"

class stream_detector_impl : public stream_detector
{
  public:
    stream_detector_impl(const std::string &model_file,          //
                         int input_height, int input_width,      //
                         int feature_height, int feature_width,  //
                         int buffer_size, bool use_f16, int gauss_kernel_size)
        : buffer_size(buffer_size),
          height(input_height),
          width(input_width),
          feature_height(feature_height),
          feature_width(feature_width),
          hwc_images(nullptr, buffer_size, height, width, 3),
          chw_images(nullptr, buffer_size, 3, height, width),
          confs(nullptr, buffer_size, n_joins, feature_height, feature_width),
          pafs(nullptr, buffer_size, n_connections * 2, feature_height,
               feature_width),
          image_stream_in(buffer_size),
          image_stream_out(buffer_size),
          feature_stream_in(buffer_size),
          feature_stream_out(buffer_size),
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
            image_stream_in.put(in_stream_t{hwc_images[i], chw_images[i]});
            feature_stream_in.put(feature_stream_t{confs[i], pafs[i]});
        }

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < filenames.size(); ++i) {
                const auto p = image_stream_in.get();
                input_image(filenames[i], height, width, p.hwc_ptr, p.chw_ptr);
                image_stream_out.put(p);
            }
        }));

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < filenames.size(); ++i) {
                const auto p = image_stream_out.get();
                const auto q = feature_stream_in.get();
                runner->execute({p.chw_ptr}, {q.heatmap_ptr, q.paf_ptr}, 1);
                image_stream_in.put(p);
                feature_stream_out.put(q);
            }
        }));

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < filenames.size(); ++i) {
                const auto q = feature_stream_out.get();
                const auto humans =
                    (*paf_process)(q.heatmap_ptr, q.paf_ptr, false);
                feature_stream_in.put(q);
                printf("got %lu humnas from %d-th image\n", humans.size(), i);
            }
        }));

        for (auto &th : ths) { th.join(); }
    }

  private:
    const int buffer_size;

    const int height;
    const int width;

    const int feature_height;
    const int feature_width;

    tensor_t<uint8_t, 4> hwc_images;
    tensor_t<float, 4> chw_images;
    tensor_t<float, 4> confs;
    tensor_t<float, 4> pafs;

    struct in_stream_t {
        uint8_t *hwc_ptr;
        float *chw_ptr;
    };

    struct feature_stream_t {
        float *heatmap_ptr;
        float *paf_ptr;
    };

    channel<in_stream_t> image_stream_in;
    channel<in_stream_t> image_stream_out;

    channel<feature_stream_t> feature_stream_in;
    channel<feature_stream_t> feature_stream_out;

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
