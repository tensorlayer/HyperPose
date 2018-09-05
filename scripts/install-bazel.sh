#!/bin/sh
set -e

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl -s https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
apt update
apt install -y bazel
