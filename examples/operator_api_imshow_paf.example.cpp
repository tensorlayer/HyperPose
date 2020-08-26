#include "utils.hpp"
#include <gflags/gflags.h>
#include <hyperpose/hyperpose.hpp>

// Model flags
DEFINE_string(model_file, "../data/models/TinyVGG-V1-HW=256x384.uff", "Path to uff model.");
DEFINE_string(input_name, "image", "The input node name of your uff model file.");
DEFINE_string(output_name_list, "outputs/conf,outputs/paf", "The output node names(maybe more than one) of your uff model file.");

DEFINE_int32(input_height, 256, "Height of input image.");
DEFINE_int32(input_width, 384, "Width of input image.");

DEFINE_bool(logging, false, "Print the logging information or not.");

DEFINE_string(input_video, "../data/media/video.avi", "The input video path.");
DEFINE_bool(camera, false, "Using the camera as input video.");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // * Input video.
    cv::VideoCapture capture;
    if (FLAGS_camera)
        capture.open(0, cv::CAP_V4L2);
    else
        cv::VideoCapture(FLAGS_input_video);

    if (!capture.isOpened())
        example_log() << "Cannot open cv::VideoCapture.\n";

    // * Create TensorRT engine.
    namespace hp = hyperpose;
    if (FLAGS_logging)
        hp::enable_logging();

    auto engine = [&] {
        using namespace hp::dnn;
        constexpr std::string_view onnx_suffix = ".onnx";
        constexpr std::string_view uff_suffix = ".uff";

        if (std::equal(onnx_suffix.crbegin(), onnx_suffix.crend(), FLAGS_model_file.crbegin()))
            return tensorrt(onnx{ FLAGS_model_file }, { FLAGS_input_width, FLAGS_input_height }, 1);

        if (std::equal(uff_suffix.crbegin(), uff_suffix.crend(), FLAGS_model_file.crbegin()))
            return tensorrt(
                uff{ FLAGS_model_file, FLAGS_input_name, split(FLAGS_output_name_list, ',') },
                { FLAGS_input_width, FLAGS_input_height },
                1);

        example_log() << "Your model file's suffix is not [.onnx | .uff]. Your model file path: " << FLAGS_model_file << '\n';
        example_log() << "Trying to be viewed as a serialized TensorRT model.\n";

        return tensorrt(tensorrt_serialized{ FLAGS_model_file }, { FLAGS_input_width, FLAGS_input_height }, 1);
    }();

    // * post-processing: Using paf.
    hp::parser::paf parser{};

    using clk_t = std::chrono::high_resolution_clock;

    example_log() << "Inference Started. Use ESC to quit.\n";

    while (capture.isOpened()) {

        cv::Mat mat;
        capture >> mat;
        if (mat.empty()) {
            example_log() << "Got empty cv::Mat\n";
            break;
        }

        auto beg = clk_t::now();

        {
            // * TensorRT Inference.
            auto feature_maps = engine.inference({ mat });

            // * Paf.
            auto poses = parser.process(feature_maps.front()[0], feature_maps.front()[1]);

            for (auto&& pose : poses)
                hp::draw_human(mat, pose);
        }

        double fps = 1000. / std::chrono::duration<double, std::milli>(clk_t::now() - beg).count();

        cv::putText(mat, "FPS: " + std::to_string(fps), { 10, 10 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 0, 255, 0 }, 2);
        cv::imshow("HyperPose Prediction", mat);

        if (cv::waitKey(1) == 27)
            break;
    }

    example_log() << "Inference Done!\n";
}