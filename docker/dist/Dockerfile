FROM centos:6

ENV LC_ALL C
ENV LANG C
ENV LANGUAGE C

RUN yum install -y \
        epel-release \
        yum-utils \
    && yum install -y \
        curl \
        hdf5 \
        hdf5-devel \
        unzip \
        wget \
        zip \
        zlib-static \
    && yum clean all

# ----------------------------------------------------------------------------
# TODO: More sustainable way of installing gcc 5.x.
# ----------------------------------------------------------------------------
# It seems like devtoolset-4 is no longer provided in SCL repository,
# but RPM packages for devtoolset-4 are still available in centos SCL
# repoitory. We directly install these packages and their dependencies
# from RPMs using yum command.
# Packages we want to install are as follows:
#   devtoolset-4-gcc
#   devtoolset-4-gcc-c++
#   devtoolset-4-gcc-gfortran
#   git19
# ----------------------------------------------------------------------------

# Installing devtoolset-4 directly from RPMs.
RUN PKGS="\
    devtoolset-4-runtime-4.1-3.el6.x86_64.rpm \
    devtoolset-4-binutils-2.25.1-8.el6.x86_64.rpm \
    devtoolset-4-gcc-5.3.1-6.1.el6.x86_64.rpm \
    devtoolset-4-libstdc++-devel-5.3.1-6.1.el6.x86_64.rpm \
    devtoolset-4-gcc-c++-5.3.1-6.1.el6.x86_64.rpm \
    devtoolset-4-libquadmath-devel-5.3.1-6.1.el6.x86_64.rpm \
    devtoolset-4-gcc-gfortran-5.3.1-6.1.el6.x86_64.rpm" \
    && for pkg in $PKGS; do yum install -y http://mirror.centos.org/centos/6/sclo/x86_64/rh/devtoolset-4/$pkg; done \
    && yum clean all

# Installing git19 directly from RPMs.
RUN REPO_BASE=http://mirror.centos.org/centos/6/sclo/x86_64/rh/git19 \
    && yum install -y \
       $REPO_BASE/git19-runtime-1.2-4.el6.x86_64.rpm \
       $REPO_BASE/git19-perl-Git-1.9.4-4.el6.1.noarch.rpm \
       $REPO_BASE/git19-git-1.9.4-4.el6.1.x86_64.rpm \
    && yum clean all

ENV PATH=/opt/rh/git19/root/usr/bin:/opt/rh/devtoolset-4/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gti19/root/usr/lib64:/opt/rh/devtoolset-4/root/usr/lib64:/opt/rh/devtoolset-4/root/usr/lib:$LD_LIBRARY_PATH
ENV PKG_CONFIG_PATH=/opt/rh/git19/root/usr/lib64/pkgconfig:${PKG_CONFIG_PATH}

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

RUN curl -L https://github.com/google/protobuf/archive/v3.5.1.tar.gz -o /tmp/protobuf-v3.5.1.tar.gz \
    && cd /tmp \
    && tar xvf protobuf-v3.5.1.tar.gz \
    && cd protobuf-3.5.1 \
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

RUN set -xe \
    ; umask 0 \
    ; for PYV in 2.7.14 3.4.5 3.5.3 3.6.2 \
    ; do PY=${PYV%.*} \
    ; echo Setup Python $PYV \
    ; conda create -y -n py$PY python=$PY \
    ; source activate py$PY \
    ; conda install scandir scikit-image virtualenv \
    ; pip install --no-cache-dir -U -r /tmp/deps/setup_requirements.txt \
    ; echo Try to install setuptools few times to avoid conda problem. \
    ; (pip install --no-cache-dir -U setuptools || pip install --no-cache-dir -U setuptools || pip install --no-cache-dir -U setuptools || true) \
    ; pip install --no-cache-dir -U -r /tmp/deps/requirements.txt \
    ; pip install --no-cache-dir -U -r /tmp/deps/test_requirements.txt \
    ; pip install --no-cache-dir -U "pytest-pep8" \
    ; pip install --no-cache-dir -U "pytest-cov" \
    ; pip install --no-cache-dir -U "Cython<0.25" \
    ; pip install --no-cache-dir -U "numpy<1.11" \
    ; source deactivate \
    ; done

