MAKEFILE ?= Makefile.config
include $(MAKEFILE)

default: libpaf

cmake_targets:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(CURDIR)

libpaf: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC)
