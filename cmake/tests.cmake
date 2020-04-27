INCLUDE(${CMAKE_SOURCE_DIR}/cmake/helpers.cmake)

ENABLE_TESTING()

FILE(GLOB_RECURSE POSE_TESTS ${CMAKE_SOURCE_DIR}/examples/tests/*.test.cpp)

foreach(TEST_FULL_PATH ${POSE_TESTS})
    MESSAGE(STATUS ">>> [TEST] TO BUILD ${TEST_FULL_PATH}")
    GET_FILENAME_COMPONENT(TEST_NAME ${TEST_FULL_PATH} NAME_WE)
    # ~ NAME_WE means filename without directory | longest extension
    # ~ See more details at https://cmake.org/cmake/help/v3.0/command/get_filename_component.html

    SET(TEST_TAR test.${TEST_NAME})
    add_executable(${TEST_TAR} ${TEST_FULL_PATH})
    set_property(
            TARGET ${TEST_TAR} PROPERTY COMPILE_FLAGS "")
    target_link_libraries(${TEST_TAR} PRIVATE helpers)
    add_test(NAME ${TEST_TAR} COMMAND ${TEST_TAR})
endforeach()