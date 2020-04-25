#pragma once

#include "../../utility/data.hpp"

namespace swiftpose
{

namespace parser
{

class paf
{
  public:
    paf(cv::Size feature_size, cv::Size image_size,
        int n_joins = 1 + COCO_N_PAIRS /* 1 + COCO_N_PARTS */,
        int n_connections = COCO_N_PAIRS /* COCO_N_PAIRS */);
    std::vector<human_t> process(feature_map_t &paf, feature_map_t &conf);

    template <typename C>
    std::vector<human_t> process(C&& feature_map_containers) {
        // 1@paf, 2@conf.
        process(feature_map_containers[0], feature_map_containers[1]);
    }

    paf(const paf& p) :
        m_feature_size(p.m_feature_size),
        m_image_size(p.m_image_size),
        m_n_joints(p.m_n_joints),
        m_n_connections(p.m_n_connections),
        m_upsample_conf(p.m_upsample_conf.shape()),
        m_upsample_paf(p.m_upsample_paf.shape()),
        m_peak_finder_ptr(std::make_unique<paf::peak_finder_impl>(
                p.m_n_joints,
                p.m_image_size.height,
                p.m_image_size.width,
                p.m_peak_finder_ptr->kernel_size())){}

    ~paf();

  private:
    cv::Size m_feature_size, m_image_size;
    int m_n_joints, m_n_connections;
    ttl::tensor<float, 3> m_upsample_conf, m_upsample_paf;
    class peak_finder_impl;
    std::unique_ptr<peak_finder_impl> m_peak_finder_ptr;
};

}  // namespace parser

}  // namespace swiftpose