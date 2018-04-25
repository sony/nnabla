# Build at NNabla root directory
# docker build -t <image name>:<tag> -f docker/docker/dev/Dockerfile .

FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ccache \
    clang-format-3.8 \
    cmake \
    curl \
    g++ \
    git \
    make \
    unzip \
    wget \
    zip \
    ca-certificates \
    bzip2 \
    libhdf5-dev \
    libarchive-dev

# Protobuf
RUN curl -L https://github.com/google/protobuf/archive/v3.1.0.tar.gz -o /tmp/protobuf-v3.1.0.tar.gz \
    && cd /tmp \
    && tar xvf protobuf-v3.1.0.tar.gz \
    && cd protobuf-3.1.0 \
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

# Conda environment for Python3 versions
RUN umask 0 && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
     && rm -rf Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/miniconda3/bin:$PATH

ADD python/setup_requirements.txt /tmp/deps/
ADD python/requirements.txt /tmp/deps/
ADD python/test_requirements.txt /tmp/deps/

# The following block will generate conda envs of Python 2.7-3.6 with nnabla dependencies.
RUN umask 0; for PY in 2.7 3.4 3.5 3.6 \
    ; do conda create -n py$PY python=$PY\
    && bash -c "source activate py$PY \
    && conda install virtualenv \
    && pip install -r /tmp/deps/setup_requirements.txt \
    && pip install -r /tmp/deps/requirements.txt \
    && pip install -r /tmp/deps/test_requirements.txt" \
    ; done
