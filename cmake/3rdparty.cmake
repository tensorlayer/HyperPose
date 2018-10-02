INCLUDE(ExternalProject)

SET(THIRDPARTY_PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

SET(STDTRACER_GIT_URL https://github.com/lgarithm/stdtracer.git
    CACHE STRING "URL for clone stdtracer")

EXTERNALPROJECT_ADD(libstdtracer
                    GIT_REPOSITORY
                    ${STDTRACER_GIT_URL}
                    GIT_TAG
                    97541fcd347644376350845083f245d43e7893dd
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0
                    -DCMAKE_CXX_FLAGS=-fPIC)

# TODO: write a cmake macro
ADD_LIBRARY(tracer STATIC)
SET_TARGET_PROPERTIES(tracer PROPERTIES LINKER_LANGUAGE CXX)
ADD_DEPENDENCIES(tracer libstdtracer)
TARGET_LINK_LIBRARIES(tracer stdtracer)

# TODO: use stdtensor
SET(STDTENSOR_GIT_URL https://github.com/lgarithm/stdtensor.git
    CACHE STRING "URL for clone stdtensor")

EXTERNALPROJECT_ADD(libstdtensor
                    GIT_REPOSITORY
                    ${STDTENSOR_GIT_URL}
                    GIT_TAG
                    dev
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${THIRDPARTY_PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0)

INCLUDE_DIRECTORIES(${THIRDPARTY_PREFIX}/include)
LINK_DIRECTORIES(${THIRDPARTY_PREFIX}/lib)
