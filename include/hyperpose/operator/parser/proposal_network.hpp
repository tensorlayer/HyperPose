#pragma once

#include <numeric>
#include <algorithm>
#include "../../utility/data.hpp"

namespace hyperpose {

namespace parser {

// The pose_proposal supports the feature map list using the following order:

// 0: conf_point         N x 18 x 12 x 12
// 1: conf_iou           N x 18 x 12 x 12
// 2: x                  N x 18 x 12 x 12
// 3: y                  N x 18 x 12 x 12
// 4: w                  N x 18 x 12 x 12
// 5: h                  N x 18 x 12 x 12
// 6: edge_confidence    N x 17 x 9 x 9 x 12 x 12

// -> Return human_t {x, y} \in [0, 1]

// * Ignore points not in the bounds.

/*
 * Step 1: Select grids.
 */

// TODO: Move to `.cpp`.

class pose_proposal {
public:
    pose_proposal(cv::Size net_resolution, float point_thresh = 0.05, float limb_thresh = 0.05, float mns_thresh = 0.3, int max_person = 32)
        :
        m_net_resolution(net_resolution),
        m_point_thresh(point_thresh),
        m_limb_thresh(limb_thresh),
        m_mns_thresh(mns_thresh),
        m_max_person(max_person) {}

    std::vector<human_t> process(
        const feature_map_t& conf_point, const feature_map_t& conf_iou,
        const feature_map_t& x, const feature_map_t& y, const feature_map_t& w, const feature_map_t& h,
        const feature_map_t& edge);

    inline std::vector<human_t> process(const std::vector<feature_map_t>& feature_map_list) {
        assert(feature_map_list.size() == 7);
        return this->process(feature_map_list[0], feature_map_list[1], feature_map_list[2], feature_map_list[3], feature_map_list[4], feature_map_list[5], feature_map_list[6]);
    }

private:
    cv::Size m_net_resolution;
    float m_point_thresh;
    float m_limb_thresh;
    float m_mns_thresh;
    int m_max_person;
};

}

} // namespace hyperpose