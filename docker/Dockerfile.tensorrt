FROM ubuntu:xenial

ARG NVIDIA_CUDA_PREFIX=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64
ARG NVIDIA_ML_PREFIX=http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/

ARG CUDA_REPO=cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
ARG ML_REPO=nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
ARG RT_REPO=nv-tensorrt-repo-ubuntu1604-cuda9.0-ga-trt4.0.1.6-20180612_1-1_amd64.deb

ADD sources.list.ustc /etc/apt/sources.list
RUN apt update && apt install -y curl && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

RUN curl -sLOJ ${NVIDIA_CUDA_PREFIX}/${CUDA_REPO} && \
    curl -sLOJ ${NVIDIA_ML_PREFIX}/${ML_REPO} && \
    dpkg -i ${CUDA_REPO} && \
    dpkg -i ${ML_REPO}

ADD ${RT_REPO} /tmp/
RUN dpkg -i /tmp/${RT_REPO} \
    && apt update

RUN apt install -y \
    libnvinfer-dev=4.1.2-1+cuda9.0  \
    cuda-cudart-dev-9-0=9.0.176-1 \
    cuda-libraries-dev-9-0

RUN apt install -y g++ cmake libopencv-dev
