#pragma once

#include <openpose_plus/utility/logging.hpp>
#include <ostream>

namespace poseplus {

const std::atomic<bool>& is_logging_enabled();
std::ostream& get_info_logger();
std::ostream& get_warning_logger();
std::ostream& get_error_logger();

template <typename... Args>
void info(const Args&... args)
{
    if (is_logging_enabled() == false)
        return;
    get_info_logger() << "[OpenPose-Plus::INFO   ] ";
    ((get_info_logger() << args), ...);
}

template <typename... Args>
void warning(const Args&... args)
{
    if (is_logging_enabled() == false)
        return;
    get_warning_logger() << "[OpenPose-Plus::WARNING] ";
    ((get_warning_logger() << args), ...);
}

template <typename... Args>
void error(const Args&... args)
{
    if (is_logging_enabled() == false)
        return;
    get_error_logger() << "[OpenPose-Plus::ERROR  ] ";
    ((get_error_logger() << args), ...);
    std::exit(-1);
}

} // namespace poseplus