# https://docs.bazel.build/versions/master/be/c-cpp.html#cc_library

cc_library(
    name = "trace",
    srcs = ["tracer.cpp"],
    hdrs = ["tracer.h"],
)

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
        "constant.h",
        "human.h",
        "paf.h",
        "pafprocess/pafprocess.h",
        "tensor.h",
    ],
    deps = [
        "trace",
        "@opencv",
    ],
)

cc_library(
    name = "fake_uff_runner",
    srcs = ["fake_uff-runner.cpp"],
    hdrs = [
        "uff-runner.h",
    ],
    deps = [
        "trace",
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
