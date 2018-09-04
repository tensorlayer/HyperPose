#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// A simple struct for tensor
template <typename T, uint8_t r> struct tensor_t {
    std::array<int, r> dims;
    std::vector<T> data;
};

class PoseDetector
{
  public:
    virtual ~PoseDetector() {}

    // non stable API
    using image_list_t = tensor_t<float, 4>;

    using detection_input_t = image_list_t;  // batch_size X height x width x 3

    using detection_result_t =
        std::tuple<image_list_t,  // batch_size x height x width x 19
                   image_list_t,  // batch_size x height x width x 19 x 2
                   image_list_t   // batch_size x height x width x 19 x 2
                   >;

    virtual detection_result_t
    get_detection_tensors(const detection_input_t &) = 0;
};

void create_pose_detector(const std::string &model_file,
                          std::unique_ptr<PoseDetector> &p);
