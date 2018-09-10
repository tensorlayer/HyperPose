# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library

cc_library(
    name = "input",
    srcs = ["input.cpp"],
    hdrs = ["input.h"],
    deps = ["@opencv"],
)

cc_library(
    name = "vis",
    srcs = ["vis.cpp"],
    hdrs = [
        "human.h",
        "tensor.h",
        "vis.h",
    ],
    deps = ["@opencv"],
)

cc_library(
    name = "paf",
    srcs = [
        "paf.cpp",
        "pafprocess/pafprocess.cpp",
    ],
    hdrs = [
        "human.h",
        "paf.h",
        "pafprocess/pafprocess.h",
        "tensor.h",
        "tracer.h",
    ],
    deps = ["@opencv"],
)

cc_library(
    name = "fake_uff_runner",
    srcs = ["fake_uff-runner.cpp"],
    hdrs = [
        "tracer.h",
        "uff-runner.h",
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
        "paf",
        "vis",
        "@opencv",
    ],
)
