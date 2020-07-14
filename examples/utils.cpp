#include "utils.hpp"

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

#if CV_MAJOR_VERSION > 3 || CV_MINOR_VERSION >= 4
#include <opencv2/core/utils/filesystem.hpp>

std::vector<cv::Mat> glob_images(const std::string& path)
{

    using namespace cv::utils::fs;
    std::vector<cv::Mat> batch;

    {
        std::vector<cv::String> img_list;
        cv::utils::fs::glob(path, "*.jpeg", img_list);
        for (auto&& file : img_list) {
            batch.push_back(cv::imread(file));
            std::cout << "Add file: " << file << " into batch.\n";
        }
    }

    {
        std::vector<cv::String> img_list;
        cv::utils::fs::glob(path, "*.png", img_list);
        for (auto&& file : img_list) {
            batch.push_back(cv::imread(file));
            std::cout << "Add file: " << file << " into batch.\n";
        }
    }

    {
        std::vector<cv::String> img_list;
        cv::utils::fs::glob(path, "*.jpg", img_list);
        for (auto&& file : img_list) {
            batch.push_back(cv::imread(file));
            std::cout << "Add file: " << file << " into batch.\n";
        }
    }

    return batch;
}
#else

// We haven't checked which filesystem to include yet
#ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

// Check for feature test macro for <filesystem>
#if defined(__cpp_lib_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0

// Check for feature test macro for <experimental/filesystem>
#elif defined(__cpp_lib_experimental_filesystem)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// We can't check if headers exist...
// Let's assume experimental to be safe
#elif !defined(__has_include)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Check if the header "<filesystem>" exists
#elif __has_include(<filesystem>)

// If we're compiling on Visual Studio and are not compiling with C++17, we need to use experimental
#ifdef _MSC_VER

// Check and include header that defines "_HAS_CXX17"
#if __has_include(<yvals_core.h>)
#include <yvals_core.h>

// Check for enabled C++17 support
#if defined(_HAS_CXX17) && _HAS_CXX17
// We're using C++17, so let's use the normal version
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#endif
#endif

// If the marco isn't defined yet, that means any of the other VS specific checks failed, so we need to use experimental
#ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1
#endif

// Not on Visual Studio. Let's use the normal version
#else // #ifdef _MSC_VER
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 0
#endif

// Check if the header "<filesystem>" exists
#elif __has_include(<experimental/filesystem>)
#define INCLUDE_STD_FILESYSTEM_EXPERIMENTAL 1

// Fail if neither header is available with a nice error message
#else
#define EXAMPLE_HAS_NO_STD_FS
#endif

// We priously determined that we need the exprimental version
#if INCLUDE_STD_FILESYSTEM_EXPERIMENTAL
// Include it
#include <experimental/filesystem>

// We need the alias from std::experimental::filesystem to std::filesystem
namespace std {
namespace filesystem = experimental::filesystem;
}

// We have a decent compiler and can use the normal version
#else
// Include it
#include <filesystem>
#endif

#include <regex>
std::vector<cv::Mat> glob_images(const std::string& path)
{
    namespace fs = std::filesystem;
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

#endif // #ifndef INCLUDE_STD_FILESYSTEM_EXPERIMENTAL

#ifdef EXAMPLE_HAS_NO_STD_FS // Last Try with <dirent.h>

#include <dirent.h>
std::vector<cv::Mat> glob_images(const std::string& path)
{
    std::regex image_regex{ R"((.*)\.(jpeg|jpg|png))" };
    std::vector<cv::Mat> batch;

    DIR* directory = opendir(path.c_str());
    struct dirent* direntStruct;

    if (directory != nullptr) {
        while (nullptr != (direntStruct = readdir(directory))) {
            auto file_name = direntStruct->d_name;
            if (std::regex_match(file_name, image_regex)) {
                std::cout << "Add file: " << file_name << " into batch.\n";
                batch.push_back(cv::imread(file_name));
            }
        }
    }
    closedir(directory);

    return batch;
}
#endif
#endif