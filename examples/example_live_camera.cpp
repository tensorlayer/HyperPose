#include <chrono>
#include <memory>
#include <thread>

#include "trace.hpp"
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <stdtensor>

using ttl::tensor_ref;

#include "channel.hpp"
#include "stream_detector.h"
#include "utils.hpp"
#include "vis.h"

// Model flags
DEFINE_string(model_file, "vgg.uff", "Path to uff model.");
DEFINE_int32(input_height, 368, "Height of input image.");
DEFINE_int32(input_width, 432, "Width of input image.");

// profiling flags
DEFINE_int32(buffer_size, 4, "Stream buffer size.");
DEFINE_int32(gauss_kernel_size, 17, "Gauss kernel size for smooth operation.");
DEFINE_bool(use_f16, false, "Use float16.");
DEFINE_bool(flip_rgb, true, "Flip RGB.");

struct camera_t {
    const int fps;

    channel<cv::Mat> &ch;

    camera_t(channel<cv::Mat> &ch, int fps = 24) : fps(fps), ch(ch) {}

    void monitor()
    {
        const int delay = 1000 / fps;
        const int height = FLAGS_input_height;
        const int width = FLAGS_input_width;

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            fprintf(stderr, "can't open camera\n");
            return;
        }
        cv::Mat frame(cv::Size(height, width), CV_8UC(3));
        for (int i = 0;; ++i) {
            cap >> frame;
            printf("#%d :: %d x %d\n", i, frame.size().height,
                   frame.size().width);
            ch.put(frame);
            cv::waitKey(delay);
        }
    }
};

struct screen_t {
    const std::string name;
    const int fps;
    const int delay;

    screen_t(const std::string &name, int fps = 24)
        : name(name), fps(fps), delay(1000 / fps)
    {
    }

    void display(const cv::Mat &frame)
    {
        cv::imshow(name, frame);
        cv::waitKey(delay);
    }
};

struct inputer : stream_detector::inputer_t {
    channel<cv::Mat> &ch;

    inputer(channel<cv::Mat> &ch) : ch(ch) {}

    bool operator()(int height, int width, uint8_t *hwc_ptr,
                    float *chw_ptr) override
    {
        const auto img = ch.get();

        cv::Mat resized_image(cv::Size(width, height), CV_8UC(3), hwc_ptr);
        cv::resize(img, resized_image, resized_image.size(), 0, 0);

        tensor_ref<uint8_t, 3> s(hwc_ptr, height, width, 3);
        tensor_ref<float, 3> t(chw_ptr, 3, height, width);
        for (int k = 0; k < 3; ++k) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    t.at(k, i, j) = s.at(i, j, k) / 255.0;
                }
            }
        }

        return true;
    }
};

struct handler : screen_t, stream_detector::handler_t {
    handler(const std::string &name) : screen_t(name) {}

    void operator()(cv::Mat &image, const std::vector<human_t> &humans) override
    {
        for (const auto &h : humans) {
            h.print();
            draw_human(image, h);
        }
        display(image);
    }
};

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // TODO: derive from model
    const int f_height = FLAGS_input_height / 8;
    const int f_width = FLAGS_input_width / 8;

    std::unique_ptr<stream_detector> sd(stream_detector::create(
        FLAGS_model_file, FLAGS_input_height, FLAGS_input_width, f_height,
        f_width, FLAGS_buffer_size, FLAGS_use_f16, FLAGS_gauss_kernel_size,
        FLAGS_flip_rgb));

    std::vector<std::thread> ths;

    channel<cv::Mat> ch(24);

    ths.push_back(std::thread([&]() {
        camera_t c1(ch);
        c1.monitor();
    }));

    ths.push_back(std::thread([&]() {
        inputer in(ch);
        handler handle("result");
        sd->run(in, handle, 1000000);
    }));

    for (auto &th : ths) { th.join(); }
    return 0;
}
