cc_library(
    name = "tracer",
    srcs = ["tracer.cpp"],
    hdrs = ["tracer.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "vis",
    srcs = ["vis.cpp"],
    hdrs = [
        "coco.h",
        "human.h",
        "tensor.h",
        "vis.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["@opencv"],
)

cc_library(
    name = "tensor",
    hdrs = ["tensor.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "paf",
    srcs = [
        "paf.cpp",
        "post-process.cpp",
    ],
    hdrs = [
        "coco.h",
        "human.h",
        "paf.h",
        "post-process.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "tensor",
        "tracer",
        "vis",
        "@opencv",
    ],
)

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
        "tracer",
    ],
)

cc_binary(
    name = "test_main",
    srcs = ["main.cpp"],
    deps = [
        "fake_uff_runner",
        "input",
        "paf",
        "tensor",
        "vis",
        "@opencv",
    ],
)
