MAKEFILE ?= Makefile.config
include $(MAKEFILE)

default: build_with_cmake
# default: docker-build-gpu

cmake_targets:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(CURDIR)

cmake_targets_with_trace:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake -DWITH_TRACE=ON $(CMAKE_FLAGS) $(CURDIR)

build_with_cmake: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC)

pack: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC) package

pack_trace: cmake_targets_with_trace
	make -C $(BUILD_DIR) -j  $(NPROC) package

# Using Docker.
CPU_TAG = openpose-plus:builder
docker-build:
	docker build --rm -t $(CPU_TAG) -f docker/Dockerfile.builder-cpu .

GPU_TAG = openpose-plus:builder-gpu
docker-build-gpu:
	docker build --rm -t $(GPU_TAG) -f docker/Dockerfile.builder-gpu .
