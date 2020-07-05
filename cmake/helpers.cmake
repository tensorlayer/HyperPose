# The same as #pragma once.
if(__CURRENT_FILE_VAR__)
    return()
endif()
set(__CURRENT_FILE_VAR__ TRUE)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(gflags REQUIRED)
FIND_PACKAGE(Threads REQUIRED)

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(CXX_FILESYSTEM_LIBRARIES stdc++fs)
else()
    set(CXX_FILESYSTEM_LIBRARIES)
endif()

# Helper Lib.
ADD_LIBRARY(helpers examples/utils.cpp)
TARGET_LINK_LIBRARIES(helpers
        Threads::Threads
        ${OpenCV_LIBS}
        ${CXX_FILESYSTEM_LIBRARIES})
ADD_GLOBAL_DEPS(helpers)
