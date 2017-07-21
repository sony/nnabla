FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc python-pip python-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip \
    && pip install -U wheel \
    && pip install -U setuptools

RUN pip install -U nnabla
