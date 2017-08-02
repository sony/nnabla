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
    bzip2

RUN wget https://github.com/google/protobuf/releases/download/v3.1.0/protoc-3.1.0-linux-x86_64.zip -O /tmp/protoc-3.1.0-linux-x86_64.zip \
    && cd /usr/local \
    && unzip /tmp/protoc-3.1.0-linux-x86_64.zip \
    && chmod 755 /usr/local/bin/protoc \
    && rm -f /tmp/protoc-3.1.0-linux-x86_64.zip readme.txt

# Conda environment for Python3 versions
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
     && rm -rf Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/miniconda3/bin:$PATH
RUN chmod -R ugo+w /opt/miniconda3

ADD python/setup_requirements.txt /tmp/deps/
ADD python/requirements.txt /tmp/deps/
ADD python/test_requirements.txt /tmp/deps/

RUN conda create -n py2.7 python=2.7 \
     && bash -c "source activate py2.7 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/setup_requirements.txt \
     && pip install -r /tmp/deps/requirements.txt \
     && pip install -r /tmp/deps/test_requirements.txt"

RUN conda create -n py3.4 python=3.4 \
     && bash -c "source activate py3.4 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/setup_requirements.txt \
     && pip install -r /tmp/deps/requirements.txt \
     && pip install -r /tmp/deps/test_requirements.txt"

RUN conda create -n py3.5 python=3.5 \
     && bash -c "source activate py3.5 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/setup_requirements.txt \
     && pip install -r /tmp/deps/requirements.txt \
     && pip install -r /tmp/deps/test_requirements.txt"

RUN conda create -n py3.6 python=3.6 \
     && bash -c "source activate py3.6 \
     && conda install virtualenv \
     && pip install -r /tmp/deps/setup_requirements.txt \
     && pip install -r /tmp/deps/requirements.txt \
     && pip install -r /tmp/deps/test_requirements.txt"
