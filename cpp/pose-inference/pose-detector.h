#pragma once

#include <memory>
#include <string>
#include <vector>

class PoseDetector
{
  public:
    virtual void detect_pose(const std::string &image_path) = 0;

    // non stable API
    using vec_t = std::vector<float>;
    using detection_result_t = std::tuple<vec_t,  // 46 x 54 x 19
                                          vec_t,  // 46 x 54 x 19 x 2
                                          vec_t   // 46 x 54 x 19 x 2
                                          >;

    virtual detection_result_t
    get_detection_tensors(const std::string &image_path) = 0;
};

void create_pose_detector(std::unique_ptr<PoseDetector> &p);
