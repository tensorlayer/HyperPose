#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "NvInfer.h"

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
  public:
    Logger() : Logger(Severity::kINFO) {}

    Logger(Severity severity)
        : reportableSeverity(severity), fs(log_file, std::ios::out)
    {
    }

    void log(Severity severity, const char *msg) override
    {
        // suppress messages with severity enum value greater than the
        // reportable
        if (severity > reportableSeverity) return;

        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            fs << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            fs << "ERROR: ";
            break;
        case Severity::kWARNING:
            fs << "WARNING: ";
            break;
        case Severity::kINFO:
            fs << "INFO: ";
            break;
        default:
            fs << "UNKNOWN: ";
            break;
        }
        fs << msg << std::endl;
    }

    Severity reportableSeverity{Severity::kWARNING};

  private:
    std::string log_file = "tensorrt.log";
    std::fstream fs;
};
