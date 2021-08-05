# Copyright 2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
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
FROM ubuntu:18.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS
ARG APT_OPTS

ARG PYTHON_VERSION_MAJOR
ARG PYTHON_VERSION_MINOR
ENV PYVERNAME=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}

RUN eval ${APT_OPTS} && apt-get update
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
