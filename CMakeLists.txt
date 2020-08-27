CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(HyperPose VERSION 2.0)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)
INCLUDE(cmake/3rdparty.cmake)

OPTION(WITH_TRACE "Build with tracing library." OFF)
OPTION(BUILD_FAKE "Build fake target without CUDA libraries." OFF)
OPTION(BUILD_CLI "Build HyperPose command line tool." ON)
OPTION(BUILD_EXAMPLES "Build examples." ON)
OPTION(BUILD_USER_CODES "Build user codes." ON)
OPTION(BUILD_TESTS "Build tests." ON)
OPTION(BUILD_LIB "Build library." ON)
OPTION(BUILD_PACKAGE "Build package." OFF)

IF(WITH_TRACE)
    MESSAGE(STATUS "[HyperPose] Trace Enabled.")
    ADD_DEFINITIONS(-DENABLE_TRACE)
ENDIF()

IF(BUILD_LIB)
    MESSAGE("[HyperPose] BUILD_LIB is ON.")
    IF(BUILD_FAKE)
        INCLUDE(cmake/hyperpose.fake.cmake)
    ELSE()
        INCLUDE(cmake/hyperpose.cmake)
    ENDIF()
    IF(BUILD_PACKAGE)
        INCLUDE(cmake/package.cmake)
    ENDIF()
ENDIF()

IF(BUILD_CLI)
    MESSAGE("[HyperPose] BUILD_CLI is ON.")
    INCLUDE(cmake/cli.cmake)
ENDIF()

IF(BUILD_EXAMPLES)
    MESSAGE(STATUS "[HyperPose] Examples Building Enabled.")
    INCLUDE(cmake/examples.cmake)
ENDIF()

IF(BUILD_USER_CODES)
    MESSAGE(STATUS "[HyperPose] User Codes Building Enabled.")
    INCLUDE(cmake/user_codes.cmake)
ENDIF()

IF(BUILD_TESTS)
    MESSAGE(STATUS "[HyperPose] Tests Building Enabled.")
    INCLUDE(cmake/tests.cmake)
    ENABLE_TESTING()
ENDIF()
