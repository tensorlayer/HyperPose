#include <opencv2/opencv.hpp>

#include "input.h"
#include "mem_buffer.h"
#include "paf.h"
#include "tracer.h"
#include "uff-runner.h"
#include "vis.h"

namespace
{
const int height = 368;
const int width = 432;
const int channel = 3;

const int f_height = height / 8;
const int f_width = width / 8;
const int n_pos = 19;

}  // namespace

void inference(bool draw_results = false)
{
    tracer_t _(__func__);

    const std::string home(std::getenv("HOME"));
    auto model_file = home + "/lg/openpose/vgg.uff";

    const auto prefix = home + "/var/data/openpose/examples/media/";
    const auto image_file = prefix + "COCO_val2014_000000000192.jpg";

    std::vector<void *> inputs(1);
    mem_buffer_t<float> image(height * width * channel);
    inputs[0] = image.data();

    std::vector<void *> outputs(2);
    mem_buffer_t<float> conf(f_height * f_width * n_pos);
    mem_buffer_t<float> paf(f_height * f_width * n_pos * 2);
    outputs[0] = conf.data();
    outputs[1] = paf.data();

    auto resized_image =
        input_image(image_file.c_str(), height, width, (float *)inputs[0]);

    std::unique_ptr<UFFRunner> runner(create_runner(model_file));
    runner->execute(inputs, outputs);

    if (draw_results) {
        tracer_t _("draw_results");

        tensor_t<float, 3> conf(outputs[0], f_height, f_width, n_pos);
        tensor_t<float, 3> paf(outputs[1], f_height, f_width, n_pos * 2);

        const auto humans = estimate_paf(conf, paf);

        std::cout << "got " << humans.size() << " humans" << std::endl;
        for (const auto &h : humans) {
            h.print();
            draw_human(resized_image, h);
        }

        int idx = 0;
        const auto name = "output" + std::to_string(idx) + ".png";
        cv::imwrite(name, resized_image);
    }
}

int main(int argc, char **argv)
{
    tracer_t _(__func__);
    inference(true);
    return 0;
}
