#include "utils.hpp"

#include <experimental/filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <regex>

std::vector<std::string> split(const std::string& text, const char sep)
{
    std::vector<std::string> lines;
    std::string line;
    std::istringstream ss(text);
    while (std::getline(ss, line, sep))
        lines.push_back(line);
    return lines;
}

std::vector<cv::Mat> glob_images(const std::string& path)
{
    namespace fs = std::experimental::filesystem;
    std::regex image_regex{ R"((.*)\.(jpeg|jpg|png))" };
    std::vector<cv::Mat> batch;
    for (auto&& file : fs::directory_iterator(path)) {
        auto file_name = file.path().string();
        if (std::regex_match(file_name, image_regex)) {
            std::cout << "Add file: " << file_name << " into batch.\n";
            batch.push_back(cv::imread(file_name));
        }
    }
    return batch;
}