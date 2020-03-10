ARG CUDA_VER=10.0
ARG CUDNN_VER=7

FROM nvidia/cuda:${CUDA_VER}-cudnn${CUDNN_VER}-runtime-ubuntu16.04

ARG PYTHON_VER=3.7
ARG CUDA_VER=10.0
ARG DALI_VER=0.18
ENV PATH /opt/miniconda3/bin:$PATH
ENV OMP_NUM_THREADS 1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    curl \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN umask 0 \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && conda install -y python=${PYTHON_VER} \
    && pip install -U setuptools \
    && conda install -y opencv jupyter

RUN umask 0 \
    && pip install nnabla-ext-cuda`echo $CUDA_VER | sed 's/\.//g'`-nccl2-ubuntu16

RUN umask 0 \
    && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/${CUDA_VER} nvidia-dali==${DALI_VER}
