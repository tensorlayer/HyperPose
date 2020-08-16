#include "utils.hpp"
#include <gflags/gflags.h>
#include <hyperpose/hyperpose.hpp>

// Model flags
DEFINE_string(model_file, "../data/models/TinyVGG-V1-HW=256x384.uff", "Path to uff model.");
DEFINE_string(input_name, "image", "The input node name of your uff model file.");
DEFINE_string(output_name_list, "outputs/conf,outputs/paf", "The output node names(maybe more than one) of your uff model file.");

DEFINE_int32(input_height, 256, "Height of input image.");
DEFINE_int32(input_width, 384, "Width of input image.");

DEFINE_int32(max_batch_size, 8, "Max batch size for inference engine to execute.");
DEFINE_bool(logging, false, "Print the logging information or not.");

DEFINE_string(input_video, "../data/media/video.avi", "Video to be processed.");
DEFINE_string(output_video, "output_video.avi", "The name of output video.");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // * Input video.
    auto capture = cv::VideoCapture(FLAGS_input_video);

    // * Output video.
    auto writer = cv::VideoWriter(
        FLAGS_output_video,
        capture.get(cv::CAP_PROP_FOURCC),
        capture.get(cv::CAP_PROP_FPS),
        cv::Size(FLAGS_input_width, FLAGS_input_height));

    // Basic Information.
    example_log() << "Input video name: " << FLAGS_input_video << std::endl;
    example_log() << "Output video name: " << FLAGS_output_video << std::endl;
    example_log() << "Input Frame: Size@" << cv::Size(capture.get(cv::CAP_PROP_FRAME_WIDTH), capture.get(cv::CAP_PROP_FRAME_HEIGHT))
                  << "Count@" << capture.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;

    // * Create TensorRT engine.
    namespace hp = hyperpose;
    if (FLAGS_logging)
        hp::enable_logging();

    auto engine = [&] {
        using namespace hp::dnn;
        constexpr std::string_view onnx_suffix = ".onnx";
        constexpr std::string_view uff_suffix = ".uff";

        if (std::equal(onnx_suffix.crbegin(), onnx_suffix.crend(), FLAGS_model_file.crbegin()))
            return tensorrt(onnx{ FLAGS_model_file }, { FLAGS_input_width, FLAGS_input_height }, FLAGS_max_batch_size);

        if (std::equal(uff_suffix.crbegin(), uff_suffix.crend(), FLAGS_model_file.crbegin()))
            return tensorrt(
                uff{ FLAGS_model_file, FLAGS_input_name, split(FLAGS_output_name_list, ',') },
                { FLAGS_input_width, FLAGS_input_height },
                FLAGS_max_batch_size);

        example_log() << "Your model file's suffix is not [.onnx | .uff]. Your model file path: " << FLAGS_model_file;
        example_log() << "Trying to be viewed as a serialized TensorRT model.";

        return tensorrt(tensorrt_serialized{ FLAGS_model_file }, { FLAGS_input_width, FLAGS_input_height }, FLAGS_max_batch_size);
    }();

    // * post-processing: Using paf.
    hp::parser::paf parser{};

    using clk_t = std::chrono::high_resolution_clock;

    size_t frame_count = 0;
    auto beg = clk_t::now();
    {
        while (capture.isOpened()) {
            std::vector<cv::Mat> batch;
            for (int i = 0; i < FLAGS_max_batch_size; ++i) {
                cv::Mat mat;
                capture >> mat;
                if (mat.empty())
                    break;
                batch.push_back(mat);
            }

            if (batch.empty())
                break;

            // * TensorRT Inference.
            auto feature_map_packets = engine.inference(batch);

            // * Paf.
            std::vector<std::vector<hp::human_t>> pose_vectors;
            pose_vectors.reserve(feature_map_packets.size());
            for (auto&& packet : feature_map_packets) {
                pose_vectors.push_back(parser.process(packet[0], packet[1]));
            }

            for (size_t i = 0; i < batch.size(); ++i) {
                cv::resize(batch[i], batch[i], { FLAGS_input_width, FLAGS_input_height });
                for (auto&& pose : pose_vectors[i])
                    hp::draw_human(batch[i], pose);
                writer << batch[i];
                ++frame_count;
            }
        }
    }
    std::cout << frame_count << " images got processed. FPS = "
              << 1000. * frame_count / std::chrono::duration<double, std::milli>(clk_t::now() - beg).count()
              << '\n';
}