#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>

#include "input.h"
#include "paf.h"
#include "tensor.h"
#include "tracer.h"
#include "uff-runner.h"
#include "vis.h"

DEFINE_string(model_file, "vgg.uff", "Path to uff model.");
DEFINE_string(image_files, "", "Comma separated list of pathes to image.");

DEFINE_int32(input_height, 368, "Height of input image.");
DEFINE_int32(input_width, 432, "Width of input image.");

namespace
{
const int channel = 3;

const int n_joins = 18 + 1;
const int n_connections = 17 + 2;

}  // namespace

void inference(const std::vector<std::string> &image_files,
               bool draw_results = false)
{
    TRACE(__func__);
    const int height = FLAGS_input_height;
    const int width = FLAGS_input_width;
    const int f_height = FLAGS_input_height / 8;
    const int f_width = FLAGS_input_width / 8;
    auto paf_process =
        create(f_height, f_width, height, width, n_joins, n_connections);

    std::unique_ptr<UFFRunner> runner;
    create_openpose_runner(FLAGS_model_file, runner);

    std::vector<void *> inputs(1);
    tensor_t<float, 3> image(nullptr, height, width, channel);
    inputs[0] = image.data();

    std::vector<void *> outputs(2);
    tensor_t<float, 3> conf(nullptr, n_joins, f_height, f_width);
    tensor_t<float, 3> paf(nullptr, n_connections * 2, f_height, f_width);
    outputs[0] = conf.data();
    outputs[1] = paf.data();

    int idx = 0;
    for (const auto &filename : image_files) {
        TRACE("inference one");
        auto resized_image =
            input_image(filename.c_str(), height, width, (float *)inputs[0]);
        {
            TRACE("run tensorRT");
            runner->execute(inputs, outputs);
        }
        const auto humans = [&]() {
            TRACE("run paf_process");
            return (*paf_process)((float *)outputs[0], (float *)outputs[1]);
        }();
        if (draw_results) {
            TRACE("draw_results");
            std::cout << "got " << humans.size() << " humans" << std::endl;
            for (const auto &h : humans) {
                h.print();
                draw_human(resized_image, h);
            }
            const auto name = "output" + std::to_string(++idx) + ".png";
            cv::imwrite(name, resized_image);
        }
    }
}

std::vector<std::string> split(const std::string &text, const char sep)
{
    std::vector<std::string> lines;
    std::string line;
    std::istringstream ss(text);
    while (std::getline(ss, line, sep)) { lines.push_back(line); }
    return lines;
}

int main(int argc, char *argv[])
{
    TRACE(__func__);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const auto image_files = split(FLAGS_image_files, ',');
    inference(image_files, true);
    return 0;
}
