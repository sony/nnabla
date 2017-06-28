FROM ubuntu:16.04

LABEL maitainer "aimof"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ccache \
        cmake \
        curl \
        g++ \
        make \
        unzip \
        git \
        python-dev \
        python-pip \
        python-setuptools \
        python-virtualenv \
    && pip install --no-cache-dir -U --ignore-installed pip \
    && curl -L https://github.com/google/protobuf/releases/download/v3.1.0/protoc-3.1.0-linux-x86_64.zip -o /tmp/protoc-3.1.0-linux-x86_64.zip \
    && unzip -d /usr/local /tmp/protoc-3.1.0-linux-x86_64.zip \
    && chmod 755 /usr/local/bin/protoc

WORKDIR /home

# nnabla
RUN git clone https://github.com/sony/nnabla \
    && mkdir /home/nnabla/build \
    && pip install -U -r /home/nnabla/python/setup_requirements.txt \
    && pip install -U -r /home/nnabla/python/requirements.txt

WORKDIR /home/nnabla/build

RUN cmake ../ \
    && make -j 16

WORKDIR /home/nnabla/build/dist

RUN pip install -U nnabla-0.9.1-cp27-cp27mu-linux_x86_64.whl

