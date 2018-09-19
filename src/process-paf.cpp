#include <cstdio>
#include <memory>
#include <string>

#include <gflags/gflags.h>

#include "paf.h"
#include "tensor.h"
#include "vis.h"

DEFINE_string(heatmap_filename, "conf.idx", "Path to heatmap tensor.");
DEFINE_string(paf_filename, "paf.idx", "Path to paf tensor.");
DEFINE_string(image_filename, "", "Path to image.");
DEFINE_string(output_filename, "output.png", "Path to output image.");

void draw_detections_resuls(const std::string &heatmap_filename,
                            const std::string &paf_filename,
                            const std::string &image_filename,
                            const std::string &output_filename)
{
    TRACE(__func__);

    const auto conf = load_3d_tensor<float>(heatmap_filename);
    debug("conf", *conf);
    const int j = conf->dims[0];
    const int height = conf->dims[1];
    const int width = conf->dims[2];

    const auto paf = load_3d_tensor<float>(paf_filename);
    debug("paf", *paf);

    auto image = cv::imread(image_filename);
    const int c = paf->dims[0];

    if (image.empty()) {
        fprintf(stderr, "using blank image\n");
        image = cv::Mat(cv::Size(8 * width, 8 * height), CV_8UC(3));
    }
    const cv::Size up_size = image.size();
    auto p = create(height, width, up_size.height, up_size.width, j, c / 2);
    const auto humans = (*p)(conf->data(), paf->data());

    for (const auto h : humans) {
        h.print();
        draw_human(image, h);
    }
    printf("saved to %s\n", output_filename.c_str());
    cv::imwrite(output_filename, image);
}

int main(int argc, char *argv[])
{
    TRACE(__func__);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    draw_detections_resuls(FLAGS_heatmap_filename, FLAGS_paf_filename,
                           FLAGS_image_filename, FLAGS_output_filename);
    return 0;
}
