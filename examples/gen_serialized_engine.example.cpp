#include "utils.hpp"
#include <gflags/gflags.h>
#include <hyperpose/hyperpose.hpp>
#include <string_view>

// Model flags
DEFINE_string(model_file, "../data/models/TinyVGG-V1-HW=256x384.uff", "Path to uff model.");

DEFINE_bool(logging, false, "Print the logging information or not.");
DEFINE_string(input_name, "image", "The input node name of your model file. (for Uff model, input/output name tags required)");
DEFINE_string(output_name_list, "outputs/conf,outputs/paf", "The output node names(maybe more than one) of your uff model file.");

DEFINE_int32(input_height, 256, "Height of input image.");
DEFINE_int32(input_width, 384, "Width of input image.");
DEFINE_int32(max_batch_size, 32, "The max batch size for the exported serialized model.");

DEFINE_string(output_model, "", "Path to output serialized model.");

int main(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // * Create TensorRT engine.
    namespace hp = hyperpose;
    if (FLAGS_logging)
        hp::enable_logging();

    hp::dnn::tensorrt engine = [&] {
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
        std::exit(1);
    }();

    const auto output_path = FLAGS_output_model.empty() ? FLAGS_model_file + ".trt" : FLAGS_output_model;
    engine.save(output_path);
}