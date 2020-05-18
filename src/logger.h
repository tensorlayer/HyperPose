#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include <NvInfer.h>

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
    std::fstream fs{"tensorrt.log", std::ios::out};

    void log_with(std::ostream &os, std::string_view prefix, std::string_view msg) const {
        os << prefix << msg << std::endl;
    }

public:
    void log(Severity severity, const char* msg) override {

        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                log_with(fs, "INTERNAL_ERROR: ", msg);
                log_with(std::cerr, "INTERNAL_ERROR: ", msg);
                break;
            case Severity::kERROR:
                log_with(fs, "ERROR: ", msg);
                log_with(std::cerr, "ERROR: ", msg);
                break;
            case Severity::kWARNING:
                log_with(fs, "WARNING: ", msg);
                break;
            case Severity::kINFO:
                log_with(fs, "INFO: ", msg);
                break;
            default:
                log_with(fs, "UNKNOWN: ", msg);
                break;
        }
    }
};
