#pragma once

#include <opencv2/opencv.hpp>
#include <tensorflow/examples/pose-inference/pose-detector.h>

#include "human.h"

void draw_human(cv::Mat &img, const Human &human);
