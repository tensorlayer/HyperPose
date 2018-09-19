MAKEFILE ?= Makefile.config
include $(MAKEFILE)

default: build_with_cmake
# default: docker-build-gpu

cmake_targets:
	mkdir -p $(BUILD_DIR)
	./configure-tf.py
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(CURDIR)

build_with_cmake: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC)

build_with_bazel:
	bazel build src/...

CPU_TAG = openpose-plus:builder
docker-build:
	docker build --rm -t $(CPU_TAG) -f docker/Dockerfile.builder-cpu .

docker-run-test: docker-build
	docker run --rm -it $(CPU_TAG) ./cmake-build/Linux/test_paf

GPU_TAG = openpose-plus:builder-gpu
docker-build-gpu:
	docker build --rm -t $(GPU_TAG) -f docker/Dockerfile.builder-gpu .
