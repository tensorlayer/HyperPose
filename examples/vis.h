#pragma once

#include <opencv2/opencv.hpp>

#include <openpose-plus/human.h>

void draw_human(cv::Mat &img, const human_t &human);
