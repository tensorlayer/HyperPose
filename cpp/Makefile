TF_ROOT = $(PWD)/tensorflow
PKG = tensorflow/examples/pose-inference

app: tensorflow
	rm -fr $(TF_ROOT)/$(PKG)
	cp -r pose-inference $(TF_ROOT)/$(PKG)
	cd $(TF_ROOT) && bazel build //$(PKG)

run: app
	$(TF_ROOT)/bazel-bin/$(PKG)/pose-inference

# Use a mirror URL if this is slow for you
TF_GIT_URL = https://github.com/tensorflow/tensorflow.git
TF_VERSION = v1.10.1

tensorflow:
	git clone $(TF_GIT_URL)
	cd tensorflow && git checkout $(TF_VERSION)
