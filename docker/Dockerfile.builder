FROM ubuntu:bionic

ADD docker/sources.list.ustc /etc/apt/sources.list
RUN apt update && \
    apt install -y curl gnupg2 && \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" > /etc/apt/sources.list.d/bazel.list && \
    curl -s https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt update && \
    apt install -y make git python python3 bazel

ADD cpp /cpp
RUN cd cpp && make
