#include <gflags/gflags.h>
#include <hyperpose/hyperpose.hpp>
#include <variant>

#include "utils.hpp"

#define kCAMERA "camera"
#define kOPERATOR "operator"
#define kSTREAM "stream"
#define kPAF "paf"
#define kPPN "ppn"

// Model Configuration.
DEFINE_string(model, "../data/models/TinyVGG-V1-HW=256x384.uff", "Path to the model.");
DEFINE_string(
    post,
    kPAF,
    "Post-processing method. (`" kPAF "` -> [Part Affine Field] or `" kPPN "` -> [Pose Proposal Network])");
DEFINE_int32(w, 384, "Width of input image.");
DEFINE_int32(h, 256, "Height of input image.");
DEFINE_int32(max_batch_size, 8, "Max batch size for inference engine to execute.");

// Execution Mode
DEFINE_bool(imshow, true, "Whether to open an imshow window.");
DEFINE_string(source, "../data/media/video.avi", "Path to your source. (the path name or `" kCAMERA "` to open your webcam)");
DEFINE_string(runtime, kOPERATOR, "Runtime setting for hyperpose. (`" kSTREAM "` or `" kOPERATOR "`)");
DEFINE_bool(keep_ratio, true, "Whether to keep the aspect ration when resizing for inference.");
DEFINE_double(alpha, 0.5, "The weight of key point visualization. (from 0 to 1)");

// Saving Config.
DEFINE_string(saving_prefix, "output", "The output media resource will be named after '$(saving_prefix)_$(ID).$(format)'.");

// System Configuration
DEFINE_bool(logging, false, "Print the logging information or not.");

namespace hp = hyperpose;

class parser_variant {
public:
    template <typename Container>
    std::vector<hp::human_t> process(Container&& feature_map_containers)
    {
        return std::visit([&feature_map_containers](auto& arg) { return arg.process(feature_map_containers); }, m_parser);
    }
    parser_variant(std::variant<hp::parser::pose_proposal, hp::parser::paf> v)
        : m_parser(std::move(v))
    {
    }

private:
    std::variant<hp::parser::pose_proposal, hp::parser::paf> m_parser;
};
//parser_variant parser{parser};

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Logging Config.
    if (FLAGS_logging) {
        cli_log() << "Internal LOGGING enabled.\n";
        hp::enable_logging();
    }

    if (FLAGS_alpha < 0 || FLAGS_alpha > 1) {
        double cl = std::clamp(FLAGS_alpha, 0., 1.);
        cli_log() << "WARNING. The flag: alpha: " << FLAGS_alpha << " out of range. Clamped to " << cl << std::endl;
        FLAGS_alpha = cl;
    }

    // Mat images if any.
    auto images = []() -> std::vector<cv::Mat> {
        constexpr auto match_suffix = [](std::string_view suffix) {
            return std::equal(suffix.crbegin(), suffix.crend(), FLAGS_source.crbegin());
        };

        if (match_suffix(".jpg") || match_suffix(".jpeg") || match_suffix(".png")) {
            cli_log() << "Source: " << FLAGS_source << ". Recognized to be an image.\n";
            return { cv::imread(FLAGS_source) };
        }

        try {
            return glob_images(FLAGS_source);
        } catch (...) {
        };

        return {};
    }();

    cv::VideoCapture cap;
    if (images.empty()) {
        if (FLAGS_source == kCAMERA) {
            cli_log() << "Source: " << FLAGS_source << ". Recognized to be a webcam.\n";
            cap.open(0, cv::CAP_V4L2);
        } else
            cap.open(FLAGS_source);
    }

    if (FLAGS_imshow && FLAGS_runtime == kSTREAM) {
        FLAGS_imshow = false;
        cli_log() << "Imshow functionality is enabled only when using operator runtime mode.\n";
    }

    if (cap.isOpened() && FLAGS_runtime == kOPERATOR) {
        FLAGS_max_batch_size = 1;
        cli_log() << "Batching is not enabled when [VideoCapture + OperatorRuntime]. Hence, set max_batch_size to 1 for better performance.\n";
    }

    if (images.empty() && !cap.isOpened()) {
        cli_log() << "ERROR: Failed to parse source: " << FLAGS_source << std::endl;
        std::exit(-1);
    }

    // Engine Config.
    auto engine = [&] {
        using namespace hp::dnn;

        cli_log() << "Configuring the TensorRT Engine:"
                  << "\n--> MODEL: " << FLAGS_model
                  << "\n--> MAX_BATCH_SIZE: " << FLAGS_max_batch_size
                  << "\n--> (HxW): " << FLAGS_h << " x " << FLAGS_w << '\n';

        constexpr std::string_view onnx_suffix = ".onnx";
        constexpr std::string_view uff_suffix = ".uff";

        if (std::equal(onnx_suffix.crbegin(), onnx_suffix.crend(), FLAGS_model.crbegin()))
            return tensorrt(onnx{ FLAGS_model }, { FLAGS_w, FLAGS_h }, FLAGS_max_batch_size, FLAGS_keep_ratio);

        if (std::equal(uff_suffix.crbegin(), uff_suffix.crend(), FLAGS_model.crbegin())) {
            cli_log()
                << "WARNING: For .uff model, the CLI program only takes 'image' as input node, and 'outputs/conf,outputs/paf' as output nodes\n";
            return tensorrt(
                uff{ FLAGS_model, "image", { "outputs/conf", "outputs/paf" } },
                { FLAGS_w, FLAGS_h },
                FLAGS_max_batch_size, FLAGS_keep_ratio);
        }

        cli_log() << "Your model file's suffix is not [.onnx | .uff]. Your model file path: " << FLAGS_model << '\n';
        cli_log() << "We assume this is a serialized TensorRT model, and we'll evaluate it in this way.\n";

        return tensorrt(tensorrt_serialized{ FLAGS_model }, { FLAGS_w, FLAGS_h }, FLAGS_max_batch_size, FLAGS_keep_ratio);
    }();
    cli_log() << "DNN engine is built.\n";

    auto parser = parser_variant{ [&engine]() -> std::variant<hp::parser::pose_proposal, hp::parser::paf> {
        if (FLAGS_post == kPAF)
            return hp::parser::paf{};

        if (FLAGS_post == kPPN)
            return hp::parser::pose_proposal(engine.input_size());

        cli_log() << "ERROR: Unknown post-processing flag: `" << FLAGS_post << "`. Use `paf` or `ppn` please.\n";
        std::exit(-1);
    }() };

    if (FLAGS_runtime != kOPERATOR and FLAGS_runtime != kSTREAM) {
        cli_log() << "WARNING: Unknown runtime flag: " << FLAGS_runtime << ". Changed this using `operator`.\n";
        FLAGS_runtime = "operator";
    }

    using clk_t = std::chrono::high_resolution_clock;

    const auto make_writer = [&cap] {
        cv::Size encoding_size = cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::string video_name = FLAGS_saving_prefix + ".avi";
        cli_log() << "Output Video Configuration:"
                  << "\n--> Filename: " << video_name
                  << "\n--> Resolution: " << encoding_size
                  << "\n--> Frame Rate: " << cap.get(cv::CAP_PROP_FPS)
                  << "\n--> Frame Count: " << cap.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;

        return cv::VideoWriter(
            video_name,
            cap.get(cv::CAP_PROP_FOURCC),
            cap.get(cv::CAP_PROP_FPS),
            encoding_size);
    };

    if (FLAGS_runtime == kOPERATOR) {
        if (images.empty()) { // For CAP.

            auto writer = make_writer();
            while (cap.isOpened()) {
                cv::Mat mat;
                cap.read(mat);

                if (mat.empty()) {
                    cli_log() << "Got empty cv::Mat ... exit\n";
                    break;
                }

                auto beg = clk_t::now();
                // * TensorRT Inference.
                auto feature_maps = engine.inference({ mat });
                // * Post-Processing.
                auto poses = parser.process(feature_maps.front());

                cv::Mat background; // Maybe used.
                if (FLAGS_alpha > 0) {
                    background = mat.clone();
                }

                for (auto&& pose : poses) {
                    if (FLAGS_keep_ratio)
                        hp::resume_ratio(pose, mat.size(), engine.input_size());
                    hp::draw_human(mat, pose);
                }

                if (FLAGS_alpha > 0) {
                    cv::addWeighted(mat, FLAGS_alpha, background, 1 - FLAGS_alpha, 0, mat);
                }

                auto prediction_time = std::chrono::duration<double, std::milli>(clk_t::now() - beg).count();
                writer << mat;

                if (FLAGS_imshow) {
                    cv::putText(
                        mat, std::to_string(prediction_time) + " ms", { 10, 10 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 0, 255, 0 }, 2, cv::LineTypes::LINE_AA);
                    cv::imshow("HyperPose Prediction", mat);

                    if (cv::waitKey(1) == 27)
                        break;
                }
            }
        } else { // For Vec<Image>.
            auto beg = clk_t::now();
            // * TensorRT Inference.
            std::vector<cv::Mat> tmp{};
            size_t counter = 0;
            while (counter != images.size()) {
                auto stride = std::min((size_t)FLAGS_max_batch_size, images.size() - counter);

                tmp.clear();
                for (size_t j = 0; j < stride; ++j)
                    tmp.push_back(images[counter + j]);

                auto feature_maps = engine.inference(tmp);

                std::vector<std::vector<hp::human_t>> pose_vectors;
                pose_vectors.reserve(feature_maps.size());
                for (auto&& packet : feature_maps)
                    pose_vectors.push_back(parser.process(packet));

                for (size_t i = 0; i < tmp.size(); ++i) {
                    cv::Mat background; // Maybe used.
                    if (FLAGS_alpha > 0) {
                        background = tmp[i].clone();
                    }

                    for (auto&& pose : pose_vectors[i]) {
                        if (FLAGS_keep_ratio)
                            hp::resume_ratio(pose, tmp[i].size(), engine.input_size());
                        hp::draw_human(tmp[i], pose);
                    }

                    if (FLAGS_alpha > 0) {
                        cv::addWeighted(tmp[i], FLAGS_alpha, background, 1 - FLAGS_alpha, 0, tmp[i]);
                    }

                    auto im_name = FLAGS_saving_prefix + '_' + std::to_string(counter++) + ".png";
                    cv::imwrite(im_name, tmp[i]);
                    cli_log() << "Wrote image to " << im_name << '\n';
                }
            }

            auto inference_time = std::chrono::duration<double, std::milli>(clk_t::now() - beg).count();

            if (images.size() == 1 && FLAGS_imshow) {
                cv::imshow("HyperPose Prediction", tmp.front());
                cv::waitKey();
            }

            std::cout << counter << " images got processed. FPS = "
                      << 1000. * counter / inference_time
                      << '\n';
        }
    } else if (FLAGS_runtime == kSTREAM) {

        auto stream = hp::make_stream(engine, parser, true, FLAGS_keep_ratio);
        auto writer = make_writer();

        auto beg = clk_t::now();

        stream.add_monitor(2000);

        stream.async() << cap;
        stream.sync() >> writer;

        auto millis = std::chrono::duration<double, std::milli>(clk_t::now() - beg).count();

        std::cout << stream.processed_num() << " images got processed in " << millis << " ms, FPS = "
                  << 1000. * stream.processed_num() / millis << '\n';
    }
}