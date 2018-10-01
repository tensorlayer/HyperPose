FROM tensorrt:snapshot

RUN apt install -y g++ cmake libopencv-dev libgflags-dev
ADD . /openpose-plus
WORKDIR /openpose-plus

RUN make build_with_cmake
