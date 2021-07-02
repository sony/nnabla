# Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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
FROM ubuntu:16.04

ARG PIP_INS_OPTS
ARG PYTHONWARNINGS
ARG CURL_OPTS
ARG WGET_OPTS

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-setuptools python3-wheel git\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install ${PIP_INS_OPTS} nnabla
RUN pip3 install ${PIP_INS_OPTS} jupyter
RUN pip3 install ${PIP_INS_OPTS} sklearn
RUN pip3 install ${PIP_INS_OPTS} imageio

RUN git clone --depth=1 https://github.com/sony/nnabla && mv nnabla/tutorial . && rm -rf nnabla
RUN git clone --depth=1 https://github.com/sony/nnabla-examples && rm -rf nnabla-examples/.git  && mv nnabla-examples tutorial

WORKDIR /tutorial

CMD ["jupyter", "notebook", "--ip=*", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token='nnabla'"]
