#include <swiftpose/utility/logging.hpp>
#include <functional>
#include "logging.hpp"

namespace swiftpose {

std::ostream* _info_ptr = nullptr;
std::ostream* _warning_ptr = nullptr;
std::ostream* _error_ptr = nullptr;


std::ostream& _info() {
    return (nullptr == _info_ptr) ? std::cout : (*_info_ptr);
}

std::ostream& _warning() {
    return (nullptr == _error_ptr) ? std::cout : (*_warning_ptr);
}

std::ostream& _error() {
    return (nullptr == _error_ptr) ? std::cerr : (*_error_ptr);
}

std::ostream& a = std::cout;

void set_info_stream(std::ostream& os) {
    _info_ptr = &os;
}

void set_warning_stream(std::ostream& os) {
    _warning_ptr = &os;
}

void set_error_stream(std::ostream& os) {
    _error_ptr = &os;
}

}