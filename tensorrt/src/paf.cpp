#include "paf.h"

#include <algorithm>
#include <array>
#include <cassert>

#include <opencv2/opencv.hpp>

#include "pafprocess/pafprocess.h"
#include "tensor.h"
#include "tracer.h"

// A simple wraper of
// https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
std::vector<Human> estimate_paf(const tensor_t<float, 3> &conf,
                                const tensor_t<float, 3> &peak,
                                const tensor_t<float, 3> &paf)
{
    tracer_t _(__func__);

    const auto p1 = peak.dims[0];
    const auto p2 = peak.dims[1];
    const auto p3 = peak.dims[2];

    const auto h1 = conf.dims[0];
    const auto h2 = conf.dims[1];
    const auto h3 = conf.dims[2];

    const auto f1 = paf.dims[0];
    const auto f2 = paf.dims[1];
    const auto f3 = paf.dims[2];

    const int n_pos = 19;
    const int height = p1;
    const int width = p2;

    printf("%d x %d\n", height, width);

    process_paf(p1, p2, p3, (float *)peak.data(),  //
                h1, h2, h3, (float *)conf.data(),  //
                f1, f2, f3, (float *)paf.data());
    const int n_humans = get_num_humans();
    std::vector<Human> humans;
    for (int human_idx = 0; human_idx < n_humans; ++human_idx) {
        Human human;
        for (int part_idx = 0; part_idx < n_pos - 1; ++part_idx) {
            const int c_idx = get_part_cid(human_idx, part_idx);
            if (c_idx < 0) { continue; }
            human.add(part_idx,
                      BodyPart(part_idx,  //
                               float(get_part_x(c_idx)) / width,
                               float(get_part_y(c_idx)) / height,
                               get_part_score(c_idx)));
        }
        if (!human.empty()) {
            human.set_scope(get_score(human_idx));
            humans.push_back(human);
        }
    }
    return humans;
}

// tf.image.resize_area
// This is the same as OpenCV's INTER_AREA.
void resize_area(const tensor_t<float, 3> &input, tensor_t<float, 3> &output)
{
    tracer_t _(__func__);

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    const int target_height = output.dims[0];
    const int target_width = output.dims[1];
    const int target_channel = output.dims[2];

    assert(channel == target_channel);

    cv::Mat input_image(cv::Size(width, height), CV_32F);
    cv::Mat output_image(cv::Size(target_width, target_height), CV_32F);

    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                input_image.at<float>(i, j) = input.at(i, j, k);
            }
        }
        cv::resize(input_image, output_image, output_image.size(), 0, 0,
                   CV_INTER_AREA);
        for (int i = 0; i < target_height; ++i) {
            for (int j = 0; j < target_width; ++j) {
                output.at(i, j, k) = output_image.at<float>(i, j);
            }
        }
    }
}

void smooth(const tensor_t<float, 3> &input, tensor_t<float, 3> output)
{
    tracer_t _(__func__);

    const int filter_size = 25;
    const float sigma = 0.3;

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    assert(height == output.dims[0]);
    assert(width == output.dims[1]);
    assert(channel == output.dims[2]);

    // TODO
    std::memcpy((void *)output.data(), input.data(),
                height * width * channel * sizeof(float));
}

void maxpool_3x3(const tensor_t<float, 3> &input, tensor_t<float, 3> output)
{
    tracer_t _(__func__);

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    assert(height == output.dims[0]);
    assert(width == output.dims[1]);
    assert(channel == output.dims[2]);

    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float max_val = input.at(i, j, k);
                for (int dx = 0; dx < 3; ++dx) {
                    for (int dy = 0; dy < 3; ++dy) {
                        const int nx = i + dx - 1;
                        const int ny = j + dy - 1;
                        if (0 <= nx && nx < height && 0 <= ny && ny < width) {
                            max_val = std::max(max_val, input.at(nx, ny, k));
                        }
                    }
                }
                output.at(i, j, k) = max_val;
            }
        }
    }
}

float delta(float x, float y, bool &flag)
{
    if (x == y) {
        flag = true;
        return x;
    }
    return 0;
}

int select_peak(const tensor_t<float, 3> &smoothed,
                const tensor_t<float, 3> &peak, tensor_t<float, 3> output)
{
    tracer_t _(__func__);

    const int height = smoothed.dims[0];
    const int width = smoothed.dims[1];
    const int channel = smoothed.dims[2];

    int tot = 0;
    for (int k = 0; k < channel; ++k) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                bool is_peak = false;
                output.at(i, j, k) =
                    delta(smoothed.at(i, j, k), peak.at(i, j, k), is_peak);
                if (is_peak) { ++tot; }
            }
        }
    }

    return tot;
}

void get_peak(const tensor_t<float, 3> &input, tensor_t<float, 3> output)
{
    tracer_t _(__func__);

    const int height = input.dims[0];
    const int width = input.dims[1];
    const int channel = input.dims[2];

    assert(height == output.dims[0]);
    assert(width == output.dims[1]);
    assert(channel == output.dims[2]);

    tensor_t<float, 3> smoothed(nullptr, height, width, channel);
    tensor_t<float, 3> pool(nullptr, height, width, channel);

    smooth(input, smoothed);
    maxpool_3x3(smoothed, pool);
    int n = select_peak(smoothed, pool, output);
    printf("%d peaks\n", n);
}

// Simplified wraper of
// https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
std::vector<Human> estimate_paf(const tensor_t<float, 3> &conf,
                                const tensor_t<float, 3> &paf)
{
    tracer_t _(__func__);

    debug("conf :: ", conf);
    debug("paf :: ", paf);

    const int height = 368;
    const int width = 432;
    const int n_pos = 19;
    // const int channel = 3;

    tensor_t<float, 3> upsample_conf(nullptr, height, width, n_pos);
    resize_area(conf, upsample_conf);
    tensor_t<float, 3> upsample_paf(nullptr, height, width, n_pos * 2);
    resize_area(paf, upsample_paf);

    tensor_t<float, 3> peaks(nullptr, height, width, n_pos);
    get_peak(upsample_conf, peaks);

    debug("upsample_conf :: ", upsample_conf);
    debug("peaks :: ", peaks);
    debug("upsample_paf :: ", upsample_paf);

    return estimate_paf(upsample_conf, peaks, upsample_paf);
}
