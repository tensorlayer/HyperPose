#!/bin/sh
set -e

cp /vagrant/sources.list.ustc /etc/apt/sources.list
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl -s https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
apt update
apt install -y g++ cmake bazel python python3 python3-pip libopencv-dev
