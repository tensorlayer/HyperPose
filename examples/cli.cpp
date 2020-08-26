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
DEFINE_string(model_file, "../data/models/TinyVGG-V1-HW=256x384.uff", "Path to uff model.");
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

// System Configuration
DEFINE_bool(logging, false, "Print the logging information or not.");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Logging Config.
    namespace hp = hyperpose;
    if (FLAGS_logging) {
        cli_log() << "Internal LOGGING enabled.";
        hp::enable_logging();
    }

    // Mat images if any.
    auto images = [] () -> std::vector<cv::Mat> {
        constexpr auto match_suffix = [] (std::string_view suffix) {
            return std::equal(suffix.crbegin(), suffix.crend(), FLAGS_source.crbegin());
        };

        if (match_suffix(".jpg") || match_suffix(".jpeg") || match_suffix(".png")) {
            cli_log() << "Source: " << FLAGS_source << ". Recognized to be an image.\n";
            return { cv::imread(FLAGS_source) };
        }

        try {
            return glob_images(FLAGS_source);
        } catch (...) {};

        return {};
    }();

    std::unique_ptr<cv::VideoCapture> cap = [&images] {
        if (!images.empty())
            return std::unique_ptr<cv::VideoCapture>{nullptr};

        if (FLAGS_source == kCAMERA) {
            cli_log() << "Source: " << FLAGS_source << ". Recognized to be a webcam.\n";
            return std::make_unique<cv::VideoCapture>(0);
        }

        return std::make_unique<cv::VideoCapture>(FLAGS_source);
    }();

    if (FLAGS_imshow && FLAGS_runtime == kSTREAM) {
        FLAGS_imshow = false;
        cli_log() << "Imshow functionality is enabled only when using operator runtime mode.\n";
    }

    if (cap != nullptr && FLAGS_runtime == kOPERATOR) {
        FLAGS_max_batch_size = 1;
        cli_log() << "Batching is not enabled when [VideoCapture + OperatorRuntime]. Hence, set max_batch_size to 1 for better performance.\n";
    }

    if (images.empty() && cap == nullptr) {
        cli_log() << "ERROR: Failed to parse source: " << FLAGS_source << std::endl;
        std::exit(-1);
    }

    // Engine Config.
    auto engine = [&] {
        using namespace hp::dnn;

        cli_log() << "Configuring the TensorRT Engine:"
                  << "\n--> MODEL: " << FLAGS_model_file
                  << "\n--> MAX_BATCH_SIZE: " << FLAGS_max_batch_size
                  << "\n--> (HxW): " << FLAGS_h << " x " << FLAGS_w << '\n';

        constexpr std::string_view onnx_suffix = ".onnx";
        constexpr std::string_view uff_suffix = ".uff";

        if (std::equal(onnx_suffix.crbegin(), onnx_suffix.crend(), FLAGS_model_file.crbegin()))
            return tensorrt(onnx{ FLAGS_model_file }, { FLAGS_w, FLAGS_h }, FLAGS_max_batch_size);

        if (std::equal(uff_suffix.crbegin(), uff_suffix.crend(), FLAGS_model_file.crbegin())) {
            cli_log()
                << "WARNING: For .uff model, the CLI program only takes 'image' as input node, and 'outputs/conf,outputs/paf' as output nodes\n";
            return tensorrt(
                uff{ FLAGS_model_file, "image", {"outputs/conf", "outputs/paf"} },
                { FLAGS_w, FLAGS_h },
                FLAGS_max_batch_size);
        }

        cli_log() << "Your model file's suffix is not [.onnx | .uff]. Your model file path: " << FLAGS_model_file << '\n';
        cli_log() << "We assume this is a serialized TensorRT model, and we'll evaluate it in this way.\n";

        return tensorrt(tensorrt_serialized{ FLAGS_model_file }, { FLAGS_w, FLAGS_h }, FLAGS_max_batch_size);
    }();
    cli_log() << "DNN engine is built.\n";

    auto parser = [&engine] () -> std::variant<hp::parser::pose_proposal, hp::parser::paf> {
        if (FLAGS_post == kPAF) {
            return hp::parser::paf{};
        }

        if (FLAGS_post == kPPN) {
            return hp::parser::pose_proposal(engine.input_size());
        }

        cli_log() << "ERROR: Unknown post-processing flag: `" << FLAGS_post << "`. Use `paf` or `ppn` please.\n";
        std::exit(-1);
    }();

    if (FLAGS_runtime != kOPERATOR and FLAGS_runtime != kSTREAM) {
        cli_log() << "WARNING: Unknown runtime flag: " << FLAGS_runtime << ". Changed this using `operator`.\n";
        FLAGS_runtime = "operator";
    }

    using clk_t = std::chrono::high_resolution_clock;

    if (FLAGS_runtime == kOPERATOR) {
        if (images.empty()) { // For CAP.
            while (cap->isOpened()) {
                cv::Mat mat;
                cap->read(mat);

                if (mat.empty()) {
                    cli_log() << "Got empty cv::Mat ... exit\n";
                    break;
                }

                auto beg = clk_t::now();
                // * TensorRT Inference.
                auto feature_maps = engine.inference({ mat });
                // * Post-Processing.
                auto poses = std::visit([&feature_maps](auto& arg){ return arg.process(feature_maps.front()); }, parser);
                for (auto&& pose : poses)
                    hp::draw_human(mat, pose);
                double fps = 1000. / std::chrono::duration<double, std::milli>(clk_t::now() - beg).count();

                cv::putText(mat, "FPS: " + std::to_string(fps), { 10, 10 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 0, 255, 0 }, 2);
                cv::imshow("HyperPose Prediction", mat);
                if (cv::waitKey(1) == 27)
                    break;
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
                    tmp.push_back(images[counter+j]);

                auto feature_maps = engine.inference(tmp);

                std::vector<std::vector<hp::human_t>> pose_vectors;
                pose_vectors.reserve(feature_maps.size());
                for (auto&& packet : feature_maps)
                    pose_vectors.push_back(std::visit([&packet](auto& arg){ return arg.process(packet); }, parser));

                for (size_t i = 0; i < tmp.size(); ++i) {
                    for (auto&& pose : pose_vectors[i])
                        hp::draw_human(tmp[i], pose);
                    auto im_name = "output_" + std::to_string(counter++) + ".png";
                    cv::imwrite(im_name, tmp[i]);
                    cli_log() << "Wrote image to " << im_name << '\n';
                }
            }

            auto inference_time = std::chrono::duration<double, std::milli>(clk_t::now() - beg).count();

            if (images.size() == 1) {
                cv::imshow("HyperPose Prediction", tmp.front());
                cv::waitKey();
            }
            std::cout << counter << " images got processed. FPS = "
                      << 1000. * counter / inference_time
                      << '\n';
        }
    } else if (FLAGS_runtime == kSTREAM) {

    }
}