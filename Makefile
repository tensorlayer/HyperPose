MAKEFILE ?= Makefile.config
include $(MAKEFILE)

default: build_with_cmake

cmake_targets:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(CURDIR)

build_with_cmake: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC)

build_with_bazel:
	bazel build src/...

TAG = openpose-plus:builder
docker-build:
	docker build --rm -t $(TAG) -f docker/Dockerfile .

docker-run-test: docker-build
	docker run --rm -it $(TAG) ./cmake-build/Linux/test_paf
