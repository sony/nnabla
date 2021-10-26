# Copyright 2018,2019,2020,2021 Sony Corporation.
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

# for nnabla>=1.5.0

FROM ubuntu:18.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS
ARG APT_OPTS

RUN eval ${APT_OPTS} \
    && apt-get update && apt-get install -y --no-install-recommends \
    clang-format-3.9 \
    make \
    git \
    python3-pip \
    python3-setuptools \
    && apt-get -yqq clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install ${PIP_INS_OPTS} -U \
    autopep8 \
    future \
    pyyaml \
    tqdm \
    setuptools \
    six \
    wheel

ENV PYTHONIOENCODING utf-8

