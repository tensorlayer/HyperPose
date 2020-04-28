#include "utils.hpp"
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>
#include <swiftpose/swiftpose.hpp>

DEFINE_string(model_file, "../data/models/hao28-600000-256x384.uff",
    "Path to uff model.");
DEFINE_int32(input_height, 256, "Height of input image.");
DEFINE_int32(input_width, 384, "Width of input image.");
DEFINE_string(input_video, "../data/media/video.avi", "Video to be processed.");
DEFINE_int32(max_batch_size, 32, "Max batch size for inference engine to execute.");
DEFINE_string(output_video, "output_video.avi", "The name of output video.");

int main()
{
    namespace sp = swiftpose;

    auto capture = cv::VideoCapture(FLAGS_input_video);

    auto writer = cv::VideoWriter(
        FLAGS_output_video,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        capture.get(cv::CAP_PROP_FPS),
        cv::Size(
            capture.get(cv::CAP_PROP_FRAME_WIDTH),
            capture.get(cv::CAP_PROP_FRAME_HEIGHT)));

    // Basic information about videos.
    swiftpose_log() << "Input video name: " << FLAGS_input_video << std::endl;
    swiftpose_log() << "Output video name: " << FLAGS_output_video << std::endl;
    swiftpose_log() << "Frame: Size@"
                    << cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT))
                    << " Count@" << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;

    // Checks.
    if (!capture.isOpened()) {
        swiftpose_log() << "Video: " << FLAGS_input_video << " cannot be opened\n";
        std::exit(-1);
    }

    sp::dnn::tensorrt engine(
        FLAGS_model_file,
        { FLAGS_input_width, FLAGS_input_height },
        "image",
        { "outputs/conf", "outputs/paf" },
        FLAGS_max_batch_size,
        sp::data_type::kFLOAT,
        1. / 255);

    sp::parser::paf parser(
        { FLAGS_input_width / 8, FLAGS_input_height / 8 },
        { FLAGS_input_width, FLAGS_input_height });

    auto stream = sp::make_stream(engine, parser);

    stream.add_monitor(2000);

    stream.async() << capture;
    stream.sync() >> writer;
}