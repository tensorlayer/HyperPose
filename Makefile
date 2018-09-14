MAKEFILE ?= Makefile.config
include $(MAKEFILE)

ifeq ($(shell uname), Darwin)
	DEFAULT_TARGET = build_with_bazel
else
	DEFAULT_TARGET = build_with_cmake
endif

WORKSPACE = $(CURDIR)/tensorrt


default: $(DEFAULT_TARGET)

cmake_targets:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(CURDIR)

build_with_cmake:
	make -C $(BUILD_DIR) -j $(NPROC)

build_with_bazel:
	cd $(WORKSPACE) && bazel build src/...
