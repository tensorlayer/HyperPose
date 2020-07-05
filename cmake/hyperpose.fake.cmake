# Library Name
set(POSE_LIB_NAME hyperpose)

FIND_PACKAGE(OpenCV)

ADD_LIBRARY(
        ${POSE_LIB_NAME} # SHARED
        src/logging.cpp
        src/fake/fake_tensorrt.cpp # FAKE
        src/fake/fake_paf.cpp # FAKE
        src/data.cpp
        src/stream.cpp
        src/thread_pool.cpp
        src/pose_proposal.cpp
        src/human.cpp)

TARGET_LINK_LIBRARIES(
        ${POSE_LIB_NAME}
        ${OpenCV_LIBS})

TARGET_INCLUDE_DIRECTORIES(${POSE_LIB_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

SET_TARGET_PROPERTIES(${POSE_LIB_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION})

INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_GLOBAL_DEPS(${POSE_LIB_NAME})
