#include "utils.hpp"
#include <gflags/gflags.h>
#include <hyperpose/hyperpose.hpp>

// Model flags
DEFINE_string(model_file, "../data/models/ppn-resnet50-V2-HW=384x384.onnx", "Path to uff model.");
DEFINE_int32(input_width, 384, "Width of input image.");
DEFINE_int32(input_height, 384, "Height of input image.");

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
        example_log() << "Cannot open cv::VideoCapture.";

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

        example_log() << "Your model file's suffix is not [.onnx | .uff]. Your model file path: " << FLAGS_model_file;
        example_log() << "Trying to be viewed as a serialized TensorRT model.";

        return tensorrt(tensorrt_serialized{ FLAGS_model_file }, { FLAGS_input_width, FLAGS_input_height }, 1);
    }();

    // * post-processing: Using Pose Proposal.
    hp::parser::pose_proposal parser{ engine.input_size() };

    using clk_t = std::chrono::high_resolution_clock;

    example_log() << "Inference Started. Use ESC to quit.";

    while (capture.isOpened()) {

        cv::Mat mat;
        capture >> mat;
        if (mat.empty()) {
            example_log() << "Got empty cv::Mat";
            break;
        }

        auto beg = clk_t::now();

        {
            // * TensorRT Inference.
            auto feature_maps = engine.inference({ mat });

            // * Post-Processing.
            auto poses = parser.process(feature_maps.front());

            for (auto&& pose : poses)
                hp::draw_human(mat, pose);
        }

        double fps = 1000. / std::chrono::duration<double, std::milli>(clk_t::now() - beg).count();

        cv::putText(mat, "FPS: " + std::to_string(fps), { 10, 10 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 0, 255, 0 }, 2);
        cv::imshow("HyperPose Prediction", mat);

        if (cv::waitKey(1) == 27)
            break;
    }

    example_log() << "Inference Done!";
}