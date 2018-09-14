# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library

cc_library(
    name = "input",
    srcs = ["input.cpp"],
    hdrs = ["input.h"],
    deps = ["@opencv"],
)

cc_library(
    name = "fake_uff_runner",
    srcs = ["fake_uff-runner.cpp"],
    hdrs = [
        "uff-runner.h",
    ],
    deps = [
        "//src/paf:tracer",
    ],
)

cc_library(
    name = "mem_buffer",
    hdrs = [
        "mem_buffer.h",
    ],
)

cc_binary(
    name = "test_main",
    srcs = ["main.cpp"],
    deps = [
        "fake_uff_runner",
        "input",
        "mem_buffer",
        "//src/paf",
        "//src/paf:tensor",
        "//src/paf:vis",
        "@opencv",
    ],
)
