#include <ostream>

namespace swiftpose {

std::ostream& get_info_logger();
std::ostream& get_warning_logger();
std::ostream& get_error_logger();

template <typename ... Args>
std::ostream& info(const Args& ... args) {
    get_info_logger() << "[SwiftPost::INFO   ] ";
    ((get_info_logger() << args), ...);
}

template <typename ... Args>
std::ostream& warning(const Args& ... args) {
    get_warning_logger() << "[SwiftPost::WARNING] ";
    ((get_warning_logger() << args), ...);
}

template <typename ... Args>
std::ostream& error(const Args& ... args) {
    get_error_logger() << "[SwiftPost::ERROR  ] ";
    ((get_error_logger() << args), ...);
    std::exit(-1);
}

}