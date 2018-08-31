#pragma once

#include <memory>
#include <string>
#include <vector>

class PoseDetector
{
  public:
    virtual ~PoseDetector() {}

    // non stable API
    using vec_t = std::vector<float>;
    using detection_input_t = vec_t;              // 368 x 432 x 3
    using detection_result_t = std::tuple<vec_t,  // 46 x 54 x 19
                                          vec_t,  // 46 x 54 x 19 x 2
                                          vec_t   // 46 x 54 x 19 x 2
                                          >;

    virtual detection_result_t
    get_detection_tensors(const detection_input_t &) = 0;
};

void create_pose_detector(const std::string &model_file,
                          std::unique_ptr<PoseDetector> &p);
