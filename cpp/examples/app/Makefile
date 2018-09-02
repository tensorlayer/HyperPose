BUILD_DIR = $(CURDIR)/cmake-build/$(shell uname)

ifeq ($(shell uname), Darwin)
	NPROC = $(shell sysctl -n hw.ncpu)
else
	NPROC = $(shell nproc)
endif

default: see_pose

CMAKE_FLAGS = \
    -DCMAKE_BUILD_TYPE=Release \

cmake_targets:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(CURDIR)

see_pose: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC)

format:
	cmake-format -i CMakeLists.txt

clean:
	rm -fr $(BUILD_DIR)
