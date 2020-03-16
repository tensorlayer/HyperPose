//
// Created by ganler on 3/8/20.
//

#pragma once

#include "utility.hpp"
#include <future>
#include <openpose-plus.hpp>

namespace pose
{

namespace pre
{

class tensorrt_processing
{
  public:
    inline tensorrt_processing(const std::string &model_path, int inp_width,
                               int inp_height, int fea_width, int fea_height,
                               int max_batch_size = 8, bool use_f16 = false,
                               bool flip_rgb = true)
        : m_inp_size(inp_width, inp_height),
          m_fea_size(fea_width, fea_height),
          m_flip_rgb(flip_rgb),
          m_max_batch_size(max_batch_size),
          m_feature_map_solver(create_pose_detection_runner(
              model_path, inp_height, inp_width, max_batch_size, use_f16))
    {
    }

    inline int max_batch_size() noexcept {
        return m_max_batch_size;
    }

    inline cv::Size input_size() noexcept {
        return m_inp_size;
    }

    inline cv::Size feature_map_size() noexcept  {
        return m_fea_size;
    }

    std::future<std::vector<internal_result_t>> async_push_batch(const batch_t&);

    std::vector<internal_result_t> sync_push_batch(const batch_t&);

  private:
    const cv::Size m_inp_size;  // w, h
    const cv::Size m_fea_size;
    const int m_max_batch_size;
    const bool m_flip_rgb;

    std::unique_ptr<pose_detection_runner> m_feature_map_solver;
};

}  // namespace pre

}  // namespace pose