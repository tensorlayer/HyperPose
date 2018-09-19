
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "input.h"
#include "paf.h"
#include "tf-runner.h"
#include "tracer.h"
#include "vis.h"

DEFINE_string(tf_model, "checkpoints/freezed", "Path to tensorflow model.");
DEFINE_string(image_filename, "", "Path to image.");
DEFINE_int32(input_height, 368, "Height of input image.");
DEFINE_int32(input_width, 432, "Width of input image.");

int main()
{
    std::unique_ptr<TFRunner> runner;
    create_openpose_runner(FLAGS_tf_model, FLAGS_input_height,
                           FLAGS_input_width, runner);
    return 0;
}
