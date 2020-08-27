INCLUDE(${CMAKE_SOURCE_DIR}/cmake/helpers.cmake)

SET(CLI hyperpose-cli)
SET(CLISRC ${CMAKE_SOURCE_DIR}/examples/cli.cpp)

MESSAGE(STATUS ">>> To build [CLI]: ${CLISRC} --> ${CLI}")
ADD_EXECUTABLE(${CLI} ${CLISRC})
TARGET_LINK_LIBRARIES(${CLI} helpers hyperpose gflags)
ADD_GLOBAL_DEPS(${CLI})