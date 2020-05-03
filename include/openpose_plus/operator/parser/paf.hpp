#pragma once

#include "../../utility/data.hpp"

namespace poseplus {

namespace parser {

    class paf {
    public:
        paf(cv::Size image_size);
        std::vector<human_t> process(feature_map_t paf, feature_map_t conf);

        template <typename C>
        std::vector<human_t> process(C&& feature_map_containers)
        {
            // 1@paf, 2@conf.
            return process(feature_map_containers[0], feature_map_containers[1]);
        }

        paf(const paf& p);

        ~paf();

    private:
        cv::Size m_image_size;

        static constexpr nullptr_t UNINITIALIZED_PTR = nullptr;
        static constexpr int UNINITIALIZED_VAL = -1;

        int m_n_joints = UNINITIALIZED_VAL, m_n_connections = UNINITIALIZED_VAL;
        cv::Size m_feature_size = { UNINITIALIZED_VAL, UNINITIALIZED_VAL };
        std::unique_ptr<ttl::tensor<float, 3>> m_upsample_conf, m_upsample_paf;
        class peak_finder_impl;
        std::unique_ptr<peak_finder_impl> m_peak_finder_ptr;
    };

} // namespace parser

} // namespace poseplus