#include "logging.hpp"
#include <atomic>

namespace hyperpose {

std::atomic<std::ostream*> _info_ptr = nullptr;
std::atomic<std::ostream*> _warning_ptr = nullptr;
std::atomic<std::ostream*> _error_ptr = nullptr;

std::atomic<bool> _enable_logging{ false };

std::ostream& get_info_logger()
{
    return (nullptr == _info_ptr) ? std::cout : (*_info_ptr);
}

std::ostream& get_warning_logger()
{
    return (nullptr == _error_ptr) ? std::cout : (*_warning_ptr);
}

std::ostream& get_error_logger()
{
    return (nullptr == _error_ptr) ? std::cerr : (*_error_ptr);
}

void set_info_stream(std::ostream& os) { _info_ptr = &os; }

void set_warning_stream(std::ostream& os) { _warning_ptr = &os; }

void set_error_stream(std::ostream& os) { _error_ptr = &os; }

const std::atomic<bool>& is_logging_enabled()
{
    return _enable_logging;
}

void enable_logging()
{
    _enable_logging = true;
}

void disable_logging()
{
    _enable_logging = false;
}

} // namespace hyperpose