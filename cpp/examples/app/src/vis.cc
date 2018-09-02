#include "vis.h"

#include <opencv2/opencv.hpp>

void tensor_summary(const std::vector<float> &t)
{
    float sum = 0;
    for (auto x : t) { sum += x; }
    printf("sum: %f, mean: %f\n", sum, sum / t.size());
}

void draw_results(const PoseDetector::detection_result_t &result)
{
    const int n_pos = 19;
    const int height = 46;
    const int width = 54;

    // 46 x 54 x n_pos
    auto [conf, pafs, peak] = result;
    tensor_summary(conf);
    tensor_summary(pafs);
    tensor_summary(peak);

    const cv::Size size(width, height);
    cv::Mat conf_map(size, CV_32F);

    int idx = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const int v = *std::max_element(conf.begin() + idx,
                                            conf.begin() + idx + n_pos);
            idx += n_pos;
            conf_map.at<float>(i, j) = v * 255.0;
        }
    }

    cv::imwrite("conf.png", conf_map);
}
