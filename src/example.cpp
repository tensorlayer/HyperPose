#include <chrono>
#include <memory>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "pose_detector.h"
#include "tracer.h"

// Model flags
DEFINE_string(model_file, "vgg.uff", "Path to uff model.");
DEFINE_int32(input_height, 368, "Height of input image.");
DEFINE_int32(input_width, 432, "Width of input image.");

// profiling flags
DEFINE_int32(batch_size, 4, "Batch size.");
DEFINE_int32(repeat, 1, "Number of repeats.");

// input flags
DEFINE_string(image_files, "", "Comma separated list of pathes to image.");

std::vector<std::string> split(const std::string &text, const char sep)
{
    std::vector<std::string> lines;
    std::string line;
    std::istringstream ss(text);
    while (std::getline(ss, line, sep)) { lines.push_back(line); }
    return lines;
}

template <typename T> std::vector<T> repeat(const std::vector<T> &v, int n)
{
    std::vector<T> u;
    for (int i = 0; i < n; ++i) {
        for (const auto &x : v) { u.push_back(x); }
    }
    return u;
}

int main(int argc, char *argv[])
{
    TRACE(__func__);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // TODO: derive from model
    const int f_height = FLAGS_input_height / 8;
    const int f_width = FLAGS_input_width / 8;

    std::unique_ptr<pose_detector> pd(create_pose_detector(
        FLAGS_model_file, FLAGS_input_height, FLAGS_input_width, f_height,
        f_width, FLAGS_batch_size));

    {
        using clock_t = std::chrono::system_clock;
        using duration_t = std::chrono::duration<double>;
        const auto t0 = clock_t::now();

        auto files = repeat(split(FLAGS_image_files, ','), FLAGS_repeat);
        pd->inference(files);

        const int n = files.size();
        const duration_t d = clock_t::now() - t0;
        double mean = d.count() / n;
        printf("// inferenced %d images of %d x %d, took %fs, mean: %fms, FPS: "
               "%f, batch "
               "size: %d\n",
               n, FLAGS_input_height, FLAGS_input_width, d.count(), mean * 1000,
               1 / mean, FLAGS_batch_size);
    }

    return 0;
}
