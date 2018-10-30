CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(openpose-plus)

SET(CMAKE_CXX_STANDARD 14)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
INCLUDE(cmake/3rdparty.cmake)

OPTION(WITH_TRACE "Build with tracing library." OFF)
OPTION(BUILD_FAKE "Build fake target without CUDA libraries." OFF)

IF(ENABLE_TRACE)
    ADD_DEFINITIONS(-DENABLE_TRACE)
ENDIF()

IF(BUILD_FAKE)
    ADD_LIBRARY(openpose-plus src/fake_openpose-plus.cpp)
ELSE()
    INCLUDE(cmake/openpose-plus.cmake)
    INCLUDE(cmake/package.cmake)
ENDIF()
INCLUDE(cmake/examples.cmake)

ENABLE_TESTING()
# TODO: add tests
