#include "utils.hpp"


#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
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

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv::utils::fs;

std::vector<cv::Mat> glob_images(const std::string& path) {
    std::vector<cv::Mat> batch;

    {
        std::vector<cv::String> img_list;
        cv::utils::fs::glob(path , "*.jpeg" , img_list);
        for (auto&& file : img_list) {
            batch.push_back(cv::imread(file));
            std::cout << "Add file: " << file << " into batch.\n";
        }
    }

    {
        std::vector<cv::String> img_list;
        cv::utils::fs::glob(path , "*.png" , img_list);
        for (auto&& file : img_list) {
            batch.push_back(cv::imread(file));
            std::cout << "Add file: " << file << " into batch.\n";
        }
    }

    {
        std::vector<cv::String> img_list;
        cv::utils::fs::glob(path , "*.jpg" , img_list);
        for (auto&& file : img_list) {
            batch.push_back(cv::imread(file));
            std::cout << "Add file: " << file << " into batch.\n";
        }
    }

    return batch;
}

