#pragma once
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string& text, const char sep);

std::vector<cv::Mat> glob_images(const std::string& path);

inline constexpr auto example_log = []() -> std::ostream& {
    std::cout << "[HyperPose::EXAMPLE] ";
    return std::cout;
};

inline constexpr auto cli_log = []() -> std::ostream& {
    std::cout << "[HyperPose::CLI] ";
    return std::cout;
};