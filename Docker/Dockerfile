# Base image
ARG PYTORCH="2.5.1"
ARG CUDA="12.4"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0+PTX 8.6+PTX"
ENV TORCH_USE_CUDA_DSA=1
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV DEBIAN_FRONTEND=noninteractive
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
# RUN rm /etc/apt/sources.list.d/cuda.list \
    # && rm /etc/apt/sources.list.d/nvidia-ml.list \
RUN apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


# Core tools
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libc6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    ninja-build \
    unzip \
    vim \
    wget \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


# Python tools
# RUN pip install --upgrade pip

RUN pip install wandb==0.18.1

# Install OpenMMLab packages
ENV FORCE_CUDA="1"

# Custom nuscenes-devkit, for compatibility with new motmetrics, newest pandas
RUN pip install git+https://github.com/TRAILab/nuscenes-devkit.git

RUN git clone https://github.com/facebookresearch/sam2.git && cd sam2 && \
    pip install -e . && \
    cd checkpoints && \
    ./download_ckpts.sh

RUN pip install "mmengine==0.10.5" "mmcv==2.1.0" "mmdet==3.2.0" "mmdet3d==1.4.0"

# install spconv
RUN pip install spconv-cu124 'numpy<2'

WORKDIR /workspace/scatr
ENV PYTHONPATH="/workspace/scatr:${PYTHONPATH}"
RUN git config --global --add safe.directory /workspace/scatr
