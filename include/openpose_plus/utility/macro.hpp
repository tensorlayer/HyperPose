#pragma once

#define SWIFTPOSE_VERSION_MAJOR 0
#define SWIFTPOSE_VERSION_MINOR 1
#define SWIFTPOSE_MINIMUM_STD_REQUIREMENT (__cplusplus > 201402L)

#if !(SWIFTPOSE_MINIMUM_STD_REQUIREMENT)
#error "SwiftPose Compile Error: C++17 required."
#endif