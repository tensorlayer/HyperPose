#pragma once
#include <vector>

#include <tensorflow/examples/pose-inference/pose-detector.h>

#include "human.h"

// A simple wraper of
// https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
std::vector<std::vector<Human>> estimate_paf(const PoseDetector::output_t &);
