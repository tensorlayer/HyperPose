#include "paf.h"

#include <array>

#include <pafprocess/pafprocess.h>

// A simple wraper of
// https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
std::vector<std::vector<Human>>
estimate_paf(const PoseDetector::detection_result_t &result)
{
    const auto [conf, peak, pafs] = result;

    const auto [n, p1, p2, p3] = peak.dims;
    const auto [_n1, h1, h2, h3] = conf.dims;
    const auto [_n2, f1, f2, f3] = pafs.dims;

    const int n_pos = 19;
    const int height = p1;
    const int width = p2;

    printf("%d x %d\n", height, width);

    std::vector<std::vector<Human>> results;
    for (int i = 0; i < n; ++i) {
        process_paf(p1, p2, p3, (float *)peak.data.data(),  //
                    h1, h2, h3, (float *)conf.data.data(),  //
                    f1, f2, f3, (float *)pafs.data.data());
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
        results.push_back(humans);
    }
    return results;
}
