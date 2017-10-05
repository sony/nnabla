FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    clang-format-3.8 \
    cmake \
    curl \
    g++ \
    git \
    make \
    python-dev \
    python-pip \
    python-setuptools \
    unzip \
    zip \
    zlib1g-dev \
    libhdf5-dev \
    libarchive-dev

ADD python/setup_requirements.txt /tmp/deps/
RUN pip install -U -r /tmp/deps/setup_requirements.txt
ADD python/requirements.txt /tmp/deps/
RUN pip install -U -r /tmp/deps/requirements.txt
ADD python/test_requirements.txt /tmp/deps/
RUN pip install -U -r /tmp/deps/test_requirements.txt

# Optioinal
RUN pip install -U ipython

RUN apt-get -yqq update \
    && apt-get -yqq install --no-install-recommends \
        build-essential \
        dh-autoreconf \
        pkg-config \
    && git clone https://github.com/google/protobuf.git /protobuf \
    && cd /protobuf \
    && git checkout v3.1.0 \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install \
    && ldconfig \
    # Cleanup
    && apt-get -yqq purge dh-autoreconf \
    && apt-get -yqq clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /protobuf /protoc-gen-doc

RUN apt-get -yqq update \
    && apt-get -yqq install --no-install-recommends \
        doxygen \
        graphviz \
        pandoc \
        plantuml \
        unzip \
    && pip install -U \
        Sphinx \
        pygments \
        recommonmark==0.4.0 \
        sphinx-rtd-theme \
        sphinxcontrib-actdiag \
        sphinxcontrib-blockdiag \
        sphinxcontrib-nwdiag \
        sphinxcontrib-plantuml \
        sphinxcontrib-seqdiag \
        sphinxcontrib-toc \
    && apt-get -yqq clean \
    && rm -rf /var/lib/apt/lists/*
