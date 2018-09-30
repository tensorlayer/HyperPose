#include <atomic>
#include <cstdint>
#include <thread>

#include <opencv2/opencv.hpp>

#include "channel.hpp"
#include "input.h"
#include "paf.h"
#include "stream_detector.h"
#include "tensor.h"
#include "tracer.h"
#include "uff-runner.h"
#include "vis.h"

struct default_inputer : stream_detector::inputer_t {
    const std::vector<std::string> filenames;
    int idx;

    default_inputer(const std::vector<std::string> &filenames)
        : filenames(filenames), idx(0)
    {
    }

    bool operator()(int height, int width, uint8_t *hwc_ptr,
                    float *chw_ptr) override
    {
        if (idx >= filenames.size()) { return false; }
        input_image(filenames[idx++], height, width, hwc_ptr, chw_ptr);
        return true;
    }
};

struct default_handler : stream_detector::handler_t {
    int idx;
    default_handler() : idx(0) {}

    void operator()(cv::Mat &image, const std::vector<human_t> &humans) override
    {
        for (const auto &h : humans) {
            h.print();
            draw_human(image, h);
        }
        const auto name = "output" + std::to_string(idx++) + ".png";
        cv::imwrite(name, image);
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
          height(input_height),
          width(input_width),
          feature_height(feature_height),
          feature_width(feature_width),
          hwc_images(nullptr, buffer_size, height, width, 3),
          chw_images(nullptr, buffer_size, 3, height, width),
          confs(nullptr, buffer_size, n_joins, feature_height, feature_width),
          pafs(nullptr, buffer_size, n_connections * 2, feature_height,
               feature_width),
          image_stream_1(buffer_size),
          image_stream_2(buffer_size),
          image_stream_3(buffer_size),
          feature_stream_1(buffer_size),
          feature_stream_2(buffer_size),
          paf_process(create_paf_processor(feature_height, feature_width,
                                           input_height, input_width, n_joins,
                                           n_connections, gauss_kernel_size))
    {
        runner.reset(
            create_openpose_runner(model_file, height, width, 1, use_f16));
    }

    void run(inputer_t &in, handler_t &handle, int count) override
    {
        TRACE(__func__);
        std::vector<std::thread> ths;

        for (int i = 0; i < buffer_size; ++i) {
            image_stream_1.put(in_stream_t{hwc_images[i], chw_images[i]});
            feature_stream_1.put(feature_stream_t{confs[i], pafs[i]});
        }

        // std::atomic<bool> done(false);

        ths.push_back(std::thread([&]() {
            for (;;) {
                const auto p = image_stream_1.get();
                if (!in(height, width, p.hwc_ptr, p.chw_ptr)) {
                    // done = true;
                    break;
                }
                image_stream_2.put(p);
            }
        }));

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < count; ++i) {
                const auto p = image_stream_2.get();
                const auto q = feature_stream_1.get();
                runner->execute({p.chw_ptr}, {q.heatmap_ptr, q.paf_ptr}, 1);
                feature_stream_2.put(q);
                image_stream_3.put(p);
            }
        }));

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < count; ++i) {
                const auto p = image_stream_3.get();
                const auto q = feature_stream_2.get();
                const auto humans =
                    (*paf_process)(q.heatmap_ptr, q.paf_ptr, false);
                printf("got %lu humnas from %d-th image\n", humans.size(), i);
                bool draw_humans = true;
                if (draw_humans) {
                    cv::Mat resized_image(cv::Size(width, height), CV_8UC(3),
                                          p.hwc_ptr);
                    handle(resized_image, humans);
                }
                feature_stream_1.put(q);
                image_stream_1.put(p);
            }
        }));

        for (auto &th : ths) { th.join(); }
    }

    void run(const std::vector<std::string> &filenames) override
    {
        default_inputer in(filenames);
        default_handler handle;
        run(in, handle, filenames.size());
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

    channel<in_stream_t> image_stream_1;
    channel<in_stream_t> image_stream_2;
    channel<in_stream_t> image_stream_3;

    channel<feature_stream_t> feature_stream_1;
    channel<feature_stream_t> feature_stream_2;

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
