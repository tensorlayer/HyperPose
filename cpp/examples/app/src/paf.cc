#include "paf.h"

#include <pafprocess/pafprocess.h>

namespace
{
void tensor_summary(const std::vector<float> &t)
{
    float sum = 0;
    for (auto x : t) { sum += x; }
    printf("sum: %f, mean: %f\n", sum, sum / t.size());
}
}  // namespace

std::vector<Human> estimate_paf(const PoseDetector::detection_result_t &result)
{
    const auto [conf, peak, pafs] = result;

    tensor_summary(conf);
    tensor_summary(peak);
    tensor_summary(pafs);

    // TODO:
    float *pp;
    float *ph;
    float *pf;

    int p1, p2, p3;
    int h1, h2, h3;
    int f1, f2, f3;
    process_paf(p1, p2, p3, pp, h1, h2, h3, ph, f1, f2, f3, pf);
    const int n = get_num_humans();

    std::vector<Human> humans;
    return humans;
}
