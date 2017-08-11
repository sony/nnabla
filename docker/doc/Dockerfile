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
    zlib1g-dev

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
        qt5-default \
    # Install protoc-gen-doc
    && git clone https://github.com/google/protobuf.git /protobuf \
    && cd /protobuf \
    && git checkout v3.1.0 \
    && ./autogen.sh \
    && ./configure \
    && make \
    && make install \
    && ldconfig \
    # Install protoc-gen-doc
    && git clone https://github.com/estan/protoc-gen-doc.git /protoc-gen-doc \
    && cd /protoc-gen-doc \
    && qmake \
    && make \
    && cp protoc-gen-doc /usr/local/bin \
    # Cleanup
    && apt-get -yqq purge dh-autoreconf \
    && apt-get -yqq clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /protobuf /protoc-gen-doc

RUN cd /tmp \
    && curl -L https://www.hdfgroup.org/package/bzip2/?wpdmdl=4300 -o hdf5-1.10.1.tar.bz2 \
    && tar xfa hdf5-1.10.1.tar.bz2 \
    && cd hdf5-* \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DHDF5_BUILD_TOOLS=OFF \
        -DHDF5_BUILD_CPP_LIB=OFF \
        .. \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/*

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
    && rm -f /usr/local/lib/libarchive*.so \
    && cd / \
    && rm -rf /tmp/*

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

