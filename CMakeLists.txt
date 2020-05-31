CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
PROJECT(hyperpose)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
INCLUDE(cmake/3rdparty.cmake)

OPTION(WITH_TRACE "Build with tracing library." OFF)
OPTION(BUILD_FAKE "Build fake target without CUDA libraries." OFF)
OPTION(ENABLE_CUDA "Build with CUDA support." ON)

IF(WITH_TRACE)
    MESSAGE(STATUS "[HyperPose] Trace Enabled.")
    ADD_DEFINITIONS(-DENABLE_TRACE)
ENDIF()

INCLUDE(cmake/hyperpose.cmake)
INCLUDE(cmake/package.cmake)

INCLUDE(cmake/helpers.cmake)
INCLUDE(cmake/examples.cmake)
INCLUDE(cmake/tests.cmake)