#include <iostream>
#include <atomic>

namespace swiftpose {

void enable_logging();
void disable_logging();

void set_info_stream(std::ostream&);
void set_warning_stream(std::ostream&);
void set_error_stream(std::ostream&);

} // namespace swiftpose