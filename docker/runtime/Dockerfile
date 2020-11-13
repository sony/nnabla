############################################################
# Miniconda3
############################################################
FROM ubuntu:18.04 as miniconda

RUN cd /tmp \
	&& apt-get update \
	&& apt-get install -y wget bzip2 openssh-client \
        && rm -rf /var/lib/apt/lists/* \
        && cd / \
        && rm -rf /tmp/*

ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh

RUN . /opt/miniconda3/bin/activate && conda install python=${PYVERNAME}
RUN . /opt/miniconda3/bin/activate && conda update -y --all
RUN . /opt/miniconda3/bin/activate && conda install pip
RUN . /opt/miniconda3/bin/activate && conda install wheel
RUN . /opt/miniconda3/bin/activate && conda install opencv || true
RUN . /opt/miniconda3/bin/activate && conda clean -y --all

ARG WHL
ADD $WHL /tmp/

RUN . /opt/miniconda3/bin/activate && pip install /tmp/$(basename $WHL)

############################################################
# Runtime image
############################################################
FROM ubuntu:18.04

RUN cd /tmp \
	&& apt-get update \
	&& apt-get install -y wget bzip2 openssh-client \
        && rm -rf /var/lib/apt/lists/* \
        && cd / \
        && rm -rf /tmp/*

COPY --from=miniconda /opt/miniconda3 /opt/miniconda3
ENV PATH /opt/miniconda3/bin:$PATH
ENV LD_LIBRARY_PATH /usr/lib64:/opt/miniconda3/lib:$LD_LIBRARY_PATH
