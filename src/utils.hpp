#pragma once
#include <string>
#include <vector>

std::vector<std::string> split(const std::string &text, const char sep)
{
    std::vector<std::string> lines;
    std::string line;
    std::istringstream ss(text);
    while (std::getline(ss, line, sep)) { lines.push_back(line); }
    return lines;
}

template <typename T> std::vector<T> repeat(const std::vector<T> &v, int n)
{
    std::vector<T> u;
    for (int i = 0; i < n; ++i) {
        for (const auto &x : v) { u.push_back(x); }
    }
    return u;
}
