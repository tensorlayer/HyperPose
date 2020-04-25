if(__CURRENT_FILE_VAR__)
    return()
endif()
set(__CURRENT_FILE_VAR__ TRUE)

FIND_PACKAGE(OpenCV)
FIND_PACKAGE(gflags)
FIND_PACKAGE(Threads REQUIRED)

if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    set(CXX_FILESYSTEM_LIBRARIES stdc++fs)
else()
    set(CXX_FILESYSTEM_LIBRARIES)
endif()

# Helper Lib.
ADD_LIBRARY(helpers
        examples/input.cpp
        examples/vis.cpp
        examples/thread_pool.cpp)
TARGET_LINK_LIBRARIES(helpers
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_imgcodecs
        Threads::Threads
        ${CXX_FILESYSTEM_LIBRARIES})
ADD_GLOBAL_DEPS(helpers)