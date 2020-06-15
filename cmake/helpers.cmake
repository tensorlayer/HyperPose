# The same as #pragma once.
if(__CURRENT_FILE_VAR__)
    return()
endif()
set(__CURRENT_FILE_VAR__ TRUE)

FIND_PACKAGE(OpenCV)
FIND_PACKAGE(gflags)
FIND_PACKAGE(Threads REQUIRED)

# Helper Lib.
ADD_LIBRARY(helpers examples/utils.cpp)
TARGET_LINK_LIBRARIES(helpers
        Threads::Threads
        ${OpenCV_LIBS})
ADD_GLOBAL_DEPS(helpers)
