#include <experimental/filesystem>
#include <gflags/gflags.h>
#include <regex>
#include <swiftpose/swiftpose.hpp>

// Model flags
DEFINE_string(model_file, "../data/models/hao28-600000-256x384.uff",
              "Path to uff model.");
DEFINE_int32(input_height, 256, "Height of input image.");
DEFINE_int32(input_width, 384, "Width of input image.");
DEFINE_string(input_folder, "../data/media", "Folder of images to inference.");

int main()
{
    namespace fs = std::experimental::filesystem;
    constexpr auto log = []() -> std::ostream & {
        std::cout << "[SwiftPose::EXAMPLE] ";
        return std::cout;
    };

    // * Collect data into batch.
    log() << "Your current path: " << fs::current_path() << '\n';
    std::regex image_regex{R"((.*)\.(jpeg|jpg|png))"};
    std::vector<cv::Mat> batch;
    for (auto &&file : fs::directory_iterator(FLAGS_input_folder)) {
        auto file_name = file.path().string();
        if (std::regex_match(file_name, image_regex)) {
            log() << "Add file: " << file_name << " into batch.\n";
            batch.push_back(cv::imread(file_name));
        }
    }

    if (batch.empty()) {
        log() << "No input images got. Exiting.\n";
        exit(-1);
    }

    log() << "Batch shape: [" << batch.size() << ", 3, " << FLAGS_input_height
          << ", " << FLAGS_input_width << "]\n";

    // * Create TensorRT engine.
    namespace sp = swiftpose;
    sp::dnn::tensorrt engine(FLAGS_model_file,
                             {FLAGS_input_width, FLAGS_input_height}, "image",
                             {"outputs/conf", "outputs/paf"}, batch.size(),
                             sp::data_type::kFLOAT, 2. / 255);
    sp::parser::paf parser({FLAGS_input_width / 8, FLAGS_input_height / 8},
                           {FLAGS_input_width, FLAGS_input_height});

    using clk_t = std::chrono::high_resolution_clock;
    auto beg = clk_t::now();
    {
        // * TensorRT Inference.
        auto feature_map_packets = engine.inference(batch);
        for (const auto &packet : feature_map_packets)
            for (const auto &feature_map : packet)
                log() << feature_map << std::endl;

        // * Paf.
        std::vector<std::vector<sp::human_t >> pose_vectors;
        pose_vectors.reserve(feature_map_packets.size());
        for (auto &&packet : feature_map_packets) {
            pose_vectors.push_back(parser.process(packet[0], packet[1]));
        }

        std::cout << batch.size() << " images got processed. FPS = "
                  << 1000. * batch.size() / std::chrono::duration<double, std::milli>(clk_t::now() - beg).count() << '\n';

        for (size_t i = 0; i < batch.size(); ++i) {
            cv::resize(batch[i], batch[i], {FLAGS_input_width, FLAGS_input_height});
            for (auto&& pose : pose_vectors[i])
                sp::draw_human(batch[i], pose);
            cv::imwrite("output_" + std::to_string(i) + ".png", batch[i]);
        }
    }
}