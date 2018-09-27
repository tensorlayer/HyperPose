#include <chrono>
#include <memory>

#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "stream_detector.h"
#include "tracer.h"
#include "utils.hpp"

// Model flags
DEFINE_string(model_file, "vgg.uff", "Path to uff model.");
DEFINE_int32(input_height, 368, "Height of input image.");
DEFINE_int32(input_width, 432, "Width of input image.");

// profiling flags
DEFINE_int32(repeat, 1, "Number of repeats.");
DEFINE_int32(batch_size, 4, "Batch size.");
DEFINE_int32(gauss_kernel_size, 17, "Gauss kernel size for smooth operation.");
DEFINE_bool(use_f16, false, "Use float16.");

// input flags
DEFINE_string(image_files, "", "Comma separated list of pathes to image.");

int main(int argc, char *argv[])
{
    TRACE(__func__);
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // TODO: derive from model
    const int f_height = FLAGS_input_height / 8;
    const int f_width = FLAGS_input_width / 8;

    std::unique_ptr<stream_detector> sd(stream_detector::create(
        FLAGS_model_file, FLAGS_input_height, FLAGS_input_width, f_height,
        f_width, FLAGS_batch_size, FLAGS_use_f16, FLAGS_gauss_kernel_size));

    sd->run();

    return 0;
}
