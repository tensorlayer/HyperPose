#include "paf.h"

#include <array>

#include <pafprocess/pafprocess.h>

// A simple wraper of
// https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
std::vector<Human> estimate_paf(const PoseDetector::detection_result_t &result)
{
    const auto [conf, peak, pafs] = result;

    const int image_height = 368;
    const int image_width = 432;
    const int n_pos = 19;

    const auto [p1, p2, p3] =
        std::array<int, 3>({image_height, image_width, n_pos});
    const auto [h1, h2, h3] =
        std::array<int, 3>({image_height, image_width, n_pos});
    const auto [f1, f2, f3] =
        std::array<int, 3>({image_height, image_width, n_pos * 2});

    process_paf(p1, p2, p3, (float *)peak.data(),  //
                h1, h2, h3, (float *)conf.data(),  //
                f1, f2, f3, (float *)pafs.data());

    const int n = get_num_humans();
    printf("got %d humans\n", n);
    std::vector<Human> humans;
    for (int human_idx = 0; human_idx < n; ++human_idx) {
        Human human;
        for (int part_idx = 0; part_idx < n_pos - 1; ++part_idx) {
            const int c_idx = get_part_cid(human_idx, part_idx);
            if (c_idx < 0) { continue; }
            human.add(part_idx,
                      BodyPart(part_idx,  //
                               float(get_part_x(c_idx)) / image_width,
                               float(get_part_y(c_idx)) / image_height,
                               get_part_score(c_idx)));
        }
        if (!human.empty()) {
            human.set_scope(get_score(human_idx));
            humans.push_back(human);
        }
    }
    return humans;
}
