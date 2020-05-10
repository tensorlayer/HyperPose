#pragma once

#define POSEPLUS_VERSION_MAJOR 2
#define POSEPLUS_VERSION_MINOR 0
#define POSEPLUS_MINIMUM_STD_REQUIREMENT (__cplusplus > 201402L)

#if !(POSEPLUS_MINIMUM_STD_REQUIREMENT)
#error "OpenPose-Plus Compile Error: C++17 required."
#endif