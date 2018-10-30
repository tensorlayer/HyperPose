INCLUDE(ExternalProject)

SET(THIRDPARTY_PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

SET(STDTRACER_GIT_URL https://github.com/lgarithm/stdtracer.git
    CACHE STRING "URL for clone stdtracer")

EXTERNALPROJECT_ADD(stdtracer-repo
                    GIT_REPOSITORY
                    ${STDTRACER_GIT_URL}
                    GIT_TAG
                    c2c7e99fbbd6ead266611c2497cc1ed88c63b46c
                    PREFIX
                    ${THIRDPARTY_PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0
                    -DCMAKE_CXX_FLAGS=-fPIC)

SET(STDTENSOR_GIT_URL https://github.com/lgarithm/stdtensor.git
    CACHE STRING "URL for clone stdtensor")

EXTERNALPROJECT_ADD(stdtensor-repo
                    GIT_REPOSITORY
                    ${STDTENSOR_GIT_URL}
                    GIT_TAG
                    62a21724014c73e57a4b693f5f05acc058d14e64
                    PREFIX
                    ${THIRDPARTY_PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0)

INCLUDE_DIRECTORIES(${THIRDPARTY_PREFIX}/include)

# a virtual target for other targets to depend on
ADD_CUSTOM_TARGET(all-external-projects)
ADD_DEPENDENCIES(all-external-projects stdtracer-repo stdtensor-repo)

LINK_DIRECTORIES(${THIRDPARTY_PREFIX}/lib)

FUNCTION(ADD_GLOBAL_DEPS target)
    ADD_DEPENDENCIES(${target} all-external-projects)
    TARGET_LINK_LIBRARIES(${target} stdtracer)
ENDFUNCTION()
