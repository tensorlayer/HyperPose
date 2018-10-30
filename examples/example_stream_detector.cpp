#include <chrono>
#include <memory>

#include "trace.hpp"
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "stream_detector.h"

#include "utils.hpp"

// Model flags
DEFINE_string(model_file, "vgg.uff", "Path to uff model.");
DEFINE_int32(input_height, 368, "Height of input image.");
DEFINE_int32(input_width, 432, "Width of input image.");

// profiling flags
DEFINE_int32(repeat, 1, "Number of repeats.");
DEFINE_int32(buffer_size, 4, "Stream buffer size.");
DEFINE_int32(gauss_kernel_size, 17, "Gauss kernel size for smooth operation.");
DEFINE_bool(use_f16, false, "Use float16.");
DEFINE_bool(flip_rgb, true, "Flip RGB.");

// input flags
DEFINE_string(image_files, "", "Comma separated list of pathes to image.");

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // TODO: derive from model
    const int f_height = FLAGS_input_height / 8;
    const int f_width = FLAGS_input_width / 8;

    const auto files = repeat(split(FLAGS_image_files, ','), FLAGS_repeat);

    std::unique_ptr<stream_detector> sd(stream_detector::create(
        FLAGS_model_file, FLAGS_input_height, FLAGS_input_width, f_height,
        f_width, FLAGS_buffer_size, FLAGS_use_f16, FLAGS_gauss_kernel_size,
        FLAGS_flip_rgb));

    {
        using clock_t = std::chrono::system_clock;
        using duration_t = std::chrono::duration<double>;
        const auto t0 = clock_t::now();

        sd->run(files);

        const int n = files.size();
        const duration_t d = clock_t::now() - t0;
        double mean = d.count() / n;
        printf("// inferenced %d images of %d x %d, took %.2fs, mean: %.2fms, "
               "FPS: %f, buffer size: %d, use f16: %d, gauss kernel size: %d\n",
               n, FLAGS_input_height, FLAGS_input_width, d.count(), mean * 1000,
               1 / mean, FLAGS_buffer_size, FLAGS_use_f16,
               FLAGS_gauss_kernel_size);
    }

    return 0;
}
