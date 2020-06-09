INCLUDE(ExternalProject)

SET(THIRDPARTY_PREFIX
    ${CMAKE_SOURCE_DIR}/3rdparty
    CACHE STRING "Where to place the 3rdparty libraries")

SET(STDTRACER_GIT_URL
    https://github.com/stdml/stdtracer.git
    CACHE STRING "URL for clone stdtracer")

EXTERNALPROJECT_ADD(
    stdtracer-repo
    LOG_DOWNLOAD ON
    LOG_INSTALL ON
    LOG_CONFIGURE ON
    GIT_REPOSITORY ${STDTRACER_GIT_URL}
    GIT_TAG v0.2.1
    PREFIX ${THIRDPARTY_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_PREFIX} -DBUILD_TESTS=0
               -DBUILD_EXAMPLES=0 -DCMAKE_CXX_FLAGS=-fPIC)

SET(STDTENSOR_GIT_URL
    https://github.com/stdml/stdtensor.git
    CACHE STRING "URL for clone stdtensor")

SET(STDTENSOR_GIT_TAG
    "v0.9.1"
    CACHE STRING "git tag for checkout stdtensor")

EXTERNALPROJECT_ADD(
    stdtensor-repo
    LOG_DOWNLOAD ON
    LOG_INSTALL ON
    LOG_CONFIGURE ON
    GIT_REPOSITORY ${STDTENSOR_GIT_URL}
    GIT_TAG ${STDTENSOR_GIT_TAG}
    PREFIX ${THIRDPARTY_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_PREFIX} -DBUILD_TESTS=0 #
               -DUSE_STRICT=0 -DBUILD_LIB=0 -DBUILD_EXAMPLES=0)

INCLUDE_DIRECTORIES(${THIRDPARTY_PREFIX}/include)

# a virtual target for other targets to depend on
ADD_CUSTOM_TARGET(all-external-projects)
ADD_DEPENDENCIES(all-external-projects stdtracer-repo stdtensor-repo)

LINK_DIRECTORIES(${THIRDPARTY_PREFIX}/lib)

ADD_LIBRARY(stdtracer src/trace.cpp)

FUNCTION(ADD_GLOBAL_DEPS target)
    ADD_DEPENDENCIES(${target} all-external-projects)
    TARGET_LINK_LIBRARIES(${target} stdtracer)
ENDFUNCTION()
