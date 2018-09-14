#pragma once
#include <string>

#include <NvInfer.h>

std::string to_string(const nvinfer1::Dims &d)
{
    std::string s;
    for (int64_t i = 0; i < d.nbDims; i++) {
        if (!s.empty()) { s += ", "; }
        s += std::to_string(d.d[i]);
    }
    return "(" + s + ")";
}

std::string to_string(const nvinfer1::DataType dtype)
{
    return std::to_string(int(dtype));
}
