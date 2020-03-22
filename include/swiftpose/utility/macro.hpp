#pragma once

#define SP_VERSION_MAJOR 0
#define SP_VERSION_MINOR 0
#define SP_MINIMUM_STD_REQUIREMENT (__cplusplus > 201402L)

#if !(SP_MINIMUM_STD_REQUIREMENT)
#error "SwiftPose Compile Error: C++17 required."
#endif