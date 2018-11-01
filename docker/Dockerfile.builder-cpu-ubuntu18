# tensorlayer/openpose-plus-builder:cpu-ubuntu18
FROM ubuntu:bionic

ADD sources.list.bionic.ustc /etc/apt/sources.list
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y g++ cmake libopencv-dev libgflags-dev git
