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
