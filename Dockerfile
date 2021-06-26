# Dockerfile

# docker build .

# Based on CUDA11.0 & CuDNN8
FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Test connection
RUN apt update --allow-unauthenticated && apt install -y wget && wget www.google.com

# Install Non-GPU Dependencies.
RUN version="7.0.0-1+cuda10.2" ; \
    apt install -y \
    libnvinfer7=${version} libnvonnxparsers7=${version} libnvparsers7=${version} \
    libnvinfer-plugin7=${version} libnvinfer-dev=${version} libnvonnxparsers-dev=${version} \
    libnvparsers-dev=${version} libnvinfer-plugin-dev=${version} python-libnvinfer=${version} \
    python3-libnvinfer=${version} && \
    apt-mark hold \
    libnvinfer7 libnvonnxparsers7 libnvparsers7 libnvinfer-plugin7 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev python-libnvinfer python3-libnvinfer

# Set apt-get to automatically retry if a package download fails
RUN echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/99AcquireRetries

# Install OpenCV Dependencies
RUN apt install -y software-properties-common || apt install -y software-properties-common && \
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && \
    APT_DEPS="git cmake libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev x264 v4l-utils python3-dev python3-pip libcanberra-gtk-module libcanberra-gtk3-module" && \
    apt install -y $APT_DEPS || apt install -y $APT_DEPS && \
    python3 -m pip install numpy

# Compile OpenCV
RUN apt install -y zip && wget https://github.com/opencv/opencv/archive/refs/tags/4.4.0.zip && unzip 4.4.0.zip && \
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

# Get models
# NOTE: if you cannot install the models due to network issues:
#   1    Manually install ONNX and UFF models through: https://drive.google.com/drive/folders/1w9EjMkrjxOmMw3Rf6fXXkiv_ge7M99jR
#   2    Put all models into `${GIT_DIR}/pre_installed_models`
#   3.1  `RUN /hyperpose/scripts/download-test-data.sh`
#   3.2  `RUN mv /hyperpose/pre_installed_models /hyperpose/data/models`
RUN for file in $(find /hyperpose/scripts -type f -iname 'download*.sh'); do sh $file; done

# Build Repo
RUN cd hyperpose && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && make -j

WORKDIR /hyperpose/build

ENTRYPOINT ["./hyperpose-cli"]
