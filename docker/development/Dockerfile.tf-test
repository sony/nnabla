# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:16.04

ENV LC_ALL C
ENV LANG C
ENV LANGUAGE C

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       bzip2 \
       ca-certificates \
       ccache \
       clang-format-3.8 \
       cmake \
       curl \
       g++ \
       git \
       libarchive-dev \
       libgoogle-glog-dev \
       libgtest-dev \
       libhdf5-dev \
       libleveldb-dev \
       liblmdb-dev \
       libsnappy-dev \
       libssl-dev \
       make \
       openssl \
       unzip \
       wget \
       zip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

################################################## libarchive
RUN cd /tmp \
    && curl -L https://www.libarchive.org/downloads/libarchive-3.3.2.tar.gz -o libarchive-3.3.2.tar.gz \
    && tar xfa libarchive-3.3.2.tar.gz \
    && mkdir libarchive-build \
    && cd libarchive-build \
    && cmake \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DENABLE_NETTLE=FALSE \
        -DENABLE_OPENSSL=FALSE \
        -DENABLE_LZO=FALSE \
        -DENABLE_LZMA=FALSE \
        -DENABLE_BZip2=FALSE \
        -DENABLE_LIBXML2=FALSE \
        -DENABLE_EXPAT=FALSE \
        -DENABLE_PCREPOSIX=FALSE \
        -DENABLE_LibGCC=FALSE \
        -DENABLE_CNG=FALSE \
        -DENABLE_TAR=FALSE \
        -DENABLE_TAR_SHARED=FALSE \
        -DENABLE_CPIO=FALSE \
        -DENABLE_CPIO_SHARED=FALSE \
        -DENABLE_CAT=FALSE \
        -DENABLE_CAT_SHARED=FALSE \
        -DENABLE_XATTR=FALSE \
        -DENABLE_ACL=FALSE \
        -DENABLE_ICONV=FALSE \
        -DENABLE_TEST=FALSE \
        ../libarchive-3.3.2 \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/*

################################################## protobuf
RUN mkdir /tmp/deps \
    && cd /tmp/deps \
    && PROTOVER=3.4.1 \
    && curl -L https://github.com/google/protobuf/archive/v${PROTOVER}.tar.gz -o protobuf-v${PROTOVER}.tar.gz \
    && tar xvf protobuf-v${PROTOVER}.tar.gz \
    && cd protobuf-${PROTOVER} \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -Dprotobuf_BUILD_TESTS=OFF \
        ../cmake \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/*

################################################## miniconda3
ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}

ADD python/setup_requirements.txt /tmp/deps/
ADD python/requirements.txt /tmp/deps/
ADD python/test_requirements.txt /tmp/deps/

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && . /opt/miniconda3/bin/activate \
    && conda create -n nnabla-build python=${PYVERNAME} \
    && conda activate nnabla-build \
    && pip install --only-binary -U -r /tmp/deps/setup_requirements.txt \
    && pip install --only-binary -U -r /tmp/deps/requirements.txt \
    && pip install --only-binary -U -r /tmp/deps/test_requirements.txt \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/envs/nnabla-build/bin:$PATH
ENV LD_LIBRARY_PATH /opt/miniconda3/envs/nnabla-build/lib:$LD_LIBRARY_PATH

RUN cd /tmp \
    && git clone https://github.com/onnx/tensorflow-onnx.git \
    && cd tensorflow-onnx \
    && python setup.py install \
    && rm -rf /tmp/tensorflow-onnx
