# Dockerfile

# docker build .

# Based on CUDA10.0 & CuDNN7
FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

# Set apt-get to automatically retry if a package download fails
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/99AcquireRetries

# apt update
RUN apt update --allow-unauthenticated

# Install Non-GPU Dependencies.
RUN version="8.0.0-1+cuda10.2" ; \
    apt install -y \
    libnvinfer8=${version} libnvonnxparsers8=${version} libnvparsers8=${version} \
    libnvinfer-plugin8=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} \
    libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} && \
    apt-mark hold \
    libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev
#    && apt install -yt python-libnvinfer=${version} python3-libnvinfer=${version} && apt-mark hold python-libnvinfer python3-libnvinfer

# Install OpenCV Dependencies
RUN apt install -y software-properties-common && \
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    APT_DEPS="git cmake wget zip libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev x264 v4l-utils python3-dev python3-pip libcanberra-gtk-module libcanberra-gtk3-module" && \
    apt install -y $APT_DEPS || apt install -y $APT_DEPS && \
    python3 -m pip install numpy

# Compile OpenCV
RUN wget https://github.com/opencv/opencv/archive/refs/tags/4.4.0.zip && unzip 4.4.0.zip && \
    cd opencv-4.4.0 && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DWITH_TBB=ON \
             -DWITH_V4L=ON \
             -DBUILD_TESTS=OFF \
             -DBUILD_OPENCV_PYTHON3=OFF && \
    make -j && make install

# Install HyperPose Dependencies
RUN apt install -y python3-dev python3-pip subversion libgflags-dev

COPY . /hyperpose

# Get models: we first see if there's existing models here. If not install it throught network.
# NOTE: if you cannot install the models due to network issues:
#   1    Manually install ONNX and UFF models through: https://drive.google.com/drive/folders/1w9EjMkrjxOmMw3Rf6fXXkiv_ge7M99jR
#   2    Put all models into `${GIT_DIR}/pre_installed_models`
#   3    Re-build this docker image.
RUN ( [ `find /hyperpose/pre_installed_models -regex '.*\.\(onnx\|uff\)' | wc -l` > 0 ] && \
    mkdir -p /hyperpose/data && mv /hyperpose/pre_installed_models/ /hyperpose/data/models ) || \
    for file in $(find /hyperpose/scripts -type f -iname 'download-*-model.sh'); do sh $file; done

# Install test data.
RUN /hyperpose/scripts/download-test-data.sh

# Build Repo
RUN cd hyperpose && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

WORKDIR /hyperpose/build

ENTRYPOINT ["./hyperpose-cli"]
