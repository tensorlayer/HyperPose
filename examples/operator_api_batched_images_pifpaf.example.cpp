#include "utils.hpp"
#include <gflags/gflags.h>
#include <hyperpose/hyperpose.hpp>
#include <string_view>

// Model flags
DEFINE_string(model_file, "../data/models/openpifpaf-resnet50.onnx", "Path to the model.");

DEFINE_bool(logging, false, "Print the logging information or not.");
DEFINE_int32(input_height, 640, "Height of input image.");
DEFINE_int32(input_width, 427, "Width of input image.");

DEFINE_string(input_folder, "../data/media", "Folder of images to inference.");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // * Collect data into batch.
    std::vector<cv::Mat> batch = glob_images(FLAGS_input_folder);

    if (batch.empty()) {
        example_log() << "No input images got. Exiting.\n";
        exit(-1);
    }

    example_log() << "Batch shape: [" << batch.size() << ", 3, " << FLAGS_input_height << ", " << FLAGS_input_width << "]\n";

    // * Create TensorRT engine.
    namespace hp = hyperpose;
    if (FLAGS_logging)
        hp::enable_logging();

    auto engine = [&] {
        using namespace hp::dnn;
        constexpr std::string_view onnx_suffix = ".onnx";
        constexpr std::string_view uff_suffix = ".uff";

        if (std::equal(onnx_suffix.crbegin(), onnx_suffix.crend(), FLAGS_model_file.crbegin()))
            return tensorrt(onnx{ FLAGS_model_file }, { FLAGS_input_width, FLAGS_input_height }, batch.size());

        example_log() << "Your model file's suffix is not [.onnx | .uff]. Your model file path: " << FLAGS_model_file;
        example_log() << "Trying to be viewed as a serialized TensorRT model.";

        return tensorrt(tensorrt_serialized{ FLAGS_model_file }, { FLAGS_input_width, FLAGS_input_height }, batch.size());
    }();

    hp::parser::pifpaf parser(engine.input_size().height, engine.input_size().width);

    using clk_t = std::chrono::high_resolution_clock;
    auto beg = clk_t::now();
    {
        // * TensorRT Inference.
        auto feature_map_packets = engine.inference(batch);
        for (const auto& packet : feature_map_packets)
            for (const auto& feature_map : packet)
                example_log() << feature_map << std::endl;

        // * Paf.
        std::vector<std::vector<hp::human_t>> pose_vectors;
        pose_vectors.reserve(feature_map_packets.size());
        for (auto&& packet : feature_map_packets) {
            pose_vectors.push_back(parser.process(packet[0], packet[1]));
        }

        std::cout << batch.size() << " images got processed. FPS = "
                  << 1000. * batch.size() / std::chrono::duration<double, std::milli>(clk_t::now() - beg).count()
                  << '\n';

        for (size_t i = 0; i < batch.size(); ++i) {
            cv::resize(batch[i], batch[i], { FLAGS_input_width, FLAGS_input_height });
            for (auto&& pose : pose_vectors[i])
                hp::draw_human(batch[i], pose);
            cv::imwrite("output_" + std::to_string(i) + ".png", batch[i]);
        }
    }
}