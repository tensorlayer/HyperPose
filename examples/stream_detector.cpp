#include "stream_detector.h"

#include <cstdint>
#include <thread>

#include <opencv2/opencv.hpp>
#include <stdtensor>

using ttl::tensor;

#include <openpose-plus.h>

#include "channel.hpp"
#include "input.h"
#include "trace.hpp"
#include "vis.h"

struct default_inputer : stream_detector::inputer_t {
    const std::vector<std::string> filenames;
    const bool flip_rgb;
    int idx;

    default_inputer(const std::vector<std::string> &filenames, bool flip_rgb)
        : filenames(filenames), flip_rgb(flip_rgb), idx(0)
    {
    }

    bool operator()(int height, int width, uint8_t *hwc_ptr,
                    float *chw_ptr) override
    {
        if (idx >= filenames.size()) { return false; }
        input_image(filenames[idx++], height, width, hwc_ptr, chw_ptr,
                    flip_rgb);
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
                         int buffer_size, bool use_f16, int gauss_kernel_size,
                         bool flip_rgb)
        : buffer_size(buffer_size),
          height(input_height),
          width(input_width),
          feature_height(feature_height),
          feature_width(feature_width),
          flip_rgb(flip_rgb),
          hwc_images(buffer_size, height, width, 3),
          chw_images(buffer_size, 3, height, width),
          confs(buffer_size, n_joins, feature_height, feature_width),
          pafs(buffer_size, n_connections * 2, feature_height, feature_width),
          image_stream_1(buffer_size),
          image_stream_2(buffer_size),
          image_stream_3(buffer_size),
          feature_stream_1(buffer_size),
          feature_stream_2(buffer_size),
          process_paf(create_paf_processor(feature_height, feature_width,
                                           input_height, input_width, n_joins,
                                           n_connections, gauss_kernel_size)),
          compute_feature_maps(create_pose_detection_runner(model_file, height,
                                                            width, 1, use_f16))
    {
    }

    void run(inputer_t &in, handler_t &handle, int count) override
    {
        TRACE_SCOPE(__func__);
        std::vector<std::thread> ths;

        for (int i = 0; i < buffer_size; ++i) {
            image_stream_1.put(
                in_stream_t{hwc_images[i].data(), chw_images[i].data()});
            feature_stream_1.put(
                feature_stream_t{confs[i].data(), pafs[i].data()});
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
                (*compute_feature_maps)({p.chw_ptr}, {q.heatmap_ptr, q.paf_ptr},
                                        1);
                feature_stream_2.put(q);
                image_stream_3.put(p);
            }
        }));

        ths.push_back(std::thread([&]() {
            for (int i = 0; i < count; ++i) {
                const auto p = image_stream_3.get();
                const auto q = feature_stream_2.get();
                const auto humans =
                    (*process_paf)(q.heatmap_ptr, q.paf_ptr, false);
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
        default_inputer in(filenames, flip_rgb);
        default_handler handle;
        run(in, handle, filenames.size());
    }

  private:
    const int buffer_size;

    const int height;
    const int width;

    const int feature_height;
    const int feature_width;

    const bool flip_rgb;

    tensor<uint8_t, 4> hwc_images;
    tensor<float, 4> chw_images;
    tensor<float, 4> confs;
    tensor<float, 4> pafs;

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

    std::unique_ptr<paf_processor> process_paf;
    std::unique_ptr<pose_detection_runner> compute_feature_maps;
};

stream_detector *stream_detector::create(const std::string &model_file,
                                         int input_height, int input_width,  //
                                         int feature_height, int feature_width,
                                         int buffer_size, bool use_f16,
                                         int gauss_kernel_size, bool flip_rgb)
{
    return new stream_detector_impl(model_file, input_height, input_width,
                                    feature_height, feature_width, buffer_size,
                                    use_f16, gauss_kernel_size, flip_rgb);
}
