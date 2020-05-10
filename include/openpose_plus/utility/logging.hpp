#pragma once

/// \file logging.hpp
/// \brief About logging configuration.
/// \author Jiawei Liu(github.com/ganler)

#include <atomic>
#include <iostream>

namespace poseplus {

/// \brief Function to enable internal logging information.
/// \note By enabling logging, you can see messages about detailed detected human numbers
/// and number human parts, which is usually used for model debugging.
void enable_logging();

/// \brief Disable internal logging.
void disable_logging();

/// \brief Use a user-defined logging stream for `info` level messages.
/// \note The default stream is `std::cout`.
/// \param s User-defined stream.
void set_info_stream(std::ostream& s);

/// \brief Use a user defined logging stream for `warning` level messages.
/// \note The default stream is `std::cout`.
/// \param s User-defined stream.
void set_warning_stream(std::ostream& s);

/// \brief Use a user defined logging stream for `error` level messages.
/// \note The default stream is `std::cerr`.
/// \param s User-defined stream.
void set_error_stream(std::ostream& s);

} // namespace poseplus