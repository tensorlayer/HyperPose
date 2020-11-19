#pragma once

#include "paf.hpp"
#include "../../utility/data.hpp"

namespace hyperpose::parser {

class pifpaf{
public:
    explicit pifpaf() = default;
    std::vector<human_t> process(const feature_map_t& pif, const feature_map_t& paf);
    template <typename C>
    std::vector<human_t> process(C&& feature_map_containers)
    {
        // 1@pif, 2@paf.
        assert(feature_map_containers.size() == 2);
        return process(feature_map_containers[0], feature_map_containers[1]);
    }
private:
    float m_keypoint_thresh = 0.001f;
};

} // namespace hyperpose