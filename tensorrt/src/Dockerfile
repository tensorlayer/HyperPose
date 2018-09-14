FROM ubuntu:xenial

RUN apt update && \
    apt install -y g++ cmake libopencv-dev
ADD . /src
WORKDIR /src
