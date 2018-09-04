TF_ROOT = $(CURDIR)/tensorflow
PKG = tensorflow/examples/pose-inference

# default: opencv_example
default: all

all: \
	libpose-inference \
	opencv_example

BAZEL_BUILD_FLAGS = \
	-c opt \
	--copt=-mavx \
	--copt=-mavx2 \
	--copt=-msse4.1 \
	--copt=-msse4.2 \
	--copt=-mfma

libpose-inference: tensorflow
	rm -fr $(TF_ROOT)/$(PKG)
	cp -r $(CURDIR)/pose-inference $(TF_ROOT)/$(PKG)
	cd $(TF_ROOT) && bazel build $(BAZEL_BUILD_FLAGS) //$(PKG):all

run: libpose-inference
	$(TF_ROOT)/bazel-bin/$(PKG)/pose-inference

# Use a mirror URL if this is slow for you
# TF_GIT_URL = https://github.com/tensorflow/tensorflow.git
TF_GIT_URL = https://gitee.com/kuroicrow/tensorflow.git

TF_VERSION = v1.10.1

tensorflow:
	git clone $(TF_GIT_URL)
	cd tensorflow && git checkout $(TF_VERSION)

query_deps:
	cd $(TF_ROOT) && bazel query 'deps(//$(PKG))'

opencv_example:
	make -C examples/app
