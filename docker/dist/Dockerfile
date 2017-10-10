FROM centos:6

RUN yum install -y yum-utils
RUN yum install -y centos-release-scl
RUN yum-config-manager --enable rhel-server-rhscl-7-rpms
RUN yum install -y devtoolset-4-gcc
RUN yum install -y devtoolset-4-gcc-c++
RUN yum install -y devtoolset-4-gcc-gfortran
RUN yum install -y devtoolset-4-binutils
RUN yum install -y git19
RUN yum install -y python27
ENV PATH=/opt/rh/git19/root/usr/bin:/opt/rh/python27/root/usr/bin:/opt/rh/devtoolset-4/root/usr/bin:$PATH 
ENV LD_LIBRARY_PATH=/opt/rh/gti19/root/usr/lib64:/opt/rh/python27/root/usr/lib64:/opt/rh/devtoolset-4/root/usr/lib64:/opt/rh/devtoolset-4/root/usr/lib:$LD_LIBRARY_PATH 
ENV PKG_CONFIG_PATH=/opt/rh/git19/root/usr/lib64/pkgconfig:/opt/rh/python27/root/usr/lib64/pkgconfig:${PKG_CONFIG_PATH}

RUN yum install -y wget curl zip unzip
RUN yum install -y zlib-static

RUN mkdir -p /tmp/deps \
    && yum install -y cmake \
    && cd /tmp/deps \
    && curl -L https://cmake.org/files/v3.9/cmake-3.9.0.tar.gz -o cmake-3.9.0.tar.gz \
    && tar xf cmake-3.9.0.tar.gz \
    && cd cmake-3.9.0 \
    && mkdir build \
    && cd build \
    && cmake -DBUILD_TESTING=FALSE .. \
    && make \
    && make install \
    && cd / \
    && rm -rf /tmp/*

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

RUN cd /tmp \
    && curl -L https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.8/hdf5-1.8.16/src/hdf5-1.8.16.tar.bz2 -o hdf5-1.8.16.tar.bz2 \
    && tar xfa hdf5-1.8.16.tar.bz2 \
    && cd hdf5-* \
    && mkdir build \
    && cd build \
    && cmake \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
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
    && cd / \
    && rm -rf /tmp/*

RUN cd /tmp \
    && wget http://nixos.org/releases/patchelf/patchelf-0.9/patchelf-0.9.tar.bz2 \
    && tar xfa patchelf-0.9.tar.bz2 \
    && cd patchelf-0.9 \
    && ./configure \
    && make \
    && make install

# Conda environment for Python3 versions
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
     && rm -rf Miniconda3-latest-Linux-x86_64.sh
ENV PATH /opt/miniconda3/bin:$PATH

ADD python/setup_requirements.txt /tmp/deps/
ADD python/requirements.txt /tmp/deps/
ADD python/test_requirements.txt /tmp/deps/

RUN set -xe ; for PYV in 2.7.14 3.4.5 3.5.3 3.6.2 \
    ; do PY=${PYV%.*} \
    ; echo Setup Python $PYV \
    ; conda create -y -n py$PY python=$PYV \
    ; source activate py$PY \
    ; conda install -y virtualenv \
    ; pip install -U -r /tmp/deps/setup_requirements.txt \
    ; echo Try to install setuptools few times to avoid conda problem. \
    ; (pip install -U setuptools || pip install -U setuptools || pip install -U setuptools || true) \
    ; pip install -U -r /tmp/deps/requirements.txt \
    ; pip install -U -r /tmp/deps/test_requirements.txt \
    ; pip install -U "Cython<0.25" \
    ; pip install -U "numpy<1.11" \
    ; source deactivate \
    ; done

RUN chmod -R ugo+w /opt/miniconda3
