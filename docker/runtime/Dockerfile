FROM ubuntu:18.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS

ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}

RUN apt-get update
RUN apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libgl1-mesa-glx \
        python${PYVERNAME} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python${PYVERNAME}-distutils || echo "skip install python-distutils" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl ${CURL_OPTS} https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python${PYVERNAME} get-pip.py ${PIP_INS_OPTS}

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYVERNAME} 0

RUN pip install ${PIP_INS_OPTS} wheel
RUN pip install ${PIP_INS_OPTS} protobuf
RUN pip install ${PIP_INS_OPTS} opencv-python || true

ARG WHL_PATH
ADD $WHL_PATH/*.whl /tmp/

RUN pip install ${PIP_INS_OPTS} /tmp/*.whl && rm -rf /tmp/*

ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH
