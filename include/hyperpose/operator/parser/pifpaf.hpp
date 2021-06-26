#pragma once

#include "../../utility/data.hpp"
#include "paf.hpp"

namespace hyperpose::parser {

class pifpaf {
public:
    inline explicit pifpaf(int h, int w, float thresh = 0.1)
        : m_net_h(h)
        , m_net_w(w)
        , m_keypoint_thresh(thresh){};
    std::vector<human_t> process(const feature_map_t& pif, const feature_map_t& paf);
    template <typename C>
    std::vector<human_t> process(C&& feature_map_containers)
    {
        // 1@pif, 2@paf.
        assert(feature_map_containers.size() == 2);
        return process(feature_map_containers[0], feature_map_containers[1]);
    }

private:
    int m_net_w, m_net_h;
    float m_keypoint_thresh;
};

} // namespace hyperpose