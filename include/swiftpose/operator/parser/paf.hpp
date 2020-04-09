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
    std::vector<human_t> process(feature_map_t &conf, feature_map_t &paf);
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