INCLUDE(${CMAKE_SOURCE_DIR}/cmake/helpers.cmake)

FILE(GLOB_RECURSE POSE_TESTS ${CMAKE_SOURCE_DIR}/examples/tests/*.test.cpp)

FOREACH(TEST_FULL_PATH ${POSE_TESTS})
    GET_FILENAME_COMPONENT(TEST_NAME ${TEST_FULL_PATH} NAME_WE)
    # ~ NAME_WE means filename without directory | longest extension ~ See more
    # details at
    # https://cmake.org/cmake/help/v3.0/command/get_filename_component.html

    SET(TEST_TAR test.${TEST_NAME})

    MESSAGE(STATUS ">>> To build [TEST]: ${TEST_FULL_PATH} --> ${TEST_TAR}")

    ADD_EXECUTABLE(${TEST_TAR} ${TEST_FULL_PATH} src/thread_pool.cpp)
    TARGET_LINK_LIBRARIES(${TEST_TAR} helpers)
    SET_PROPERTY(TARGET ${TEST_TAR} PROPERTY COMPILE_FLAGS "")
    ADD_TEST(NAME ${TEST_TAR} COMMAND ${TEST_TAR})
ENDFOREACH()
