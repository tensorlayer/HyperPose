#include <opencv2/opencv.hpp>

#include "input.h"
#include "mem_buffer.h"
#include "paf/paf.h"
#include "paf/tensor.h"
#include "paf/tracer.h"
#include "paf/vis.h"
#include "uff-runner.h"

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
    TRACE(__func__);
    auto paf_process = create(f_height, f_width, height, width, n_pos, n_pos);

    const std::string home(std::getenv("HOME"));
    auto model_file = home + "/lg/openpose/vgg.uff";
    std::unique_ptr<UFFRunner> runner(create_runner(model_file));

    std::vector<void *> inputs(1);
    mem_buffer_t<float> image(height * width * channel);
    inputs[0] = image.data();

    std::vector<void *> outputs(2);
    mem_buffer_t<float> conf(f_height * f_width * n_pos);
    mem_buffer_t<float> paf(f_height * f_width * n_pos * 2);
    outputs[0] = conf.data();
    outputs[1] = paf.data();

    {
        TRACE("inference one");
        const auto prefix = home + "/var/data/openpose/examples/media/";
        const auto image_file = prefix + "COCO_val2014_000000000192.jpg";

        auto resized_image =
            input_image(image_file.c_str(), height, width, (float *)inputs[0]);

        runner->execute(inputs, outputs);

        if (draw_results) {
            TRACE("draw_results");
            const auto humans =
                (*paf_process)((float *)outputs[0], (float *)outputs[1]);

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
}

int main(int argc, char **argv)
{
    TRACE(__func__);
    inference(true);
    return 0;
}
