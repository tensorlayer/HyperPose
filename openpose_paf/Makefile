MAKEFILE ?= Makefile.config
include $(MAKEFILE)

default: libpaf

cmake_targets:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR); cmake $(CMAKE_FLAGS) $(CURDIR)

libpaf: cmake_targets
	make -C $(BUILD_DIR) -j $(NPROC)

TAG = libopenpose-pdf:builder

docker:
	docker build --rm -t $(TAG) .

docker-run:
	docker run --rm -it $(TAG)
