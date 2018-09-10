#pragma once
#include <vector>

// #include <tensorflow/examples/pose-inference/pose-detector.h>

#include "human.h"
#include "tensor.h"

// A simple wraper of
// https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
std::vector<Human> estimate_paf(const tensor_t<float, 3> &conf,
                                // TODO: calculate peak from conf
                                const tensor_t<float, 3> &peak,
                                const tensor_t<float, 3> &paf);

// Simplified wraper of
// https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
std::vector<Human> estimate_paf(const tensor_t<float, 3> &conf,
                                const tensor_t<float, 3> &paf);
