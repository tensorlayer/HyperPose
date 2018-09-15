FROM ubuntu:xenial

RUN apt update && \
    apt install -y g++ cmake libopencv-dev libgflags-dev
ADD . /openpose-plus
WORKDIR /openpose-plus
RUN make build_with_cmake
RUN curl -sLOJ https://github.com/tensorlayer/fast-openpose/files/2378505/network-outputs.gz && \
    gzip -d network-outputs.gz && \
    tar -xf network-outputs
