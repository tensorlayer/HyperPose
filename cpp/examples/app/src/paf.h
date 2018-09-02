#pragma once
#include <vector>

#include <tensorflow/examples/pose-inference/pose-detector.h>

#include "human.h"

std::vector<Human> estimate_paf(const PoseDetector::detection_result_t &);
