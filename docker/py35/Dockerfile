FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN umask 0 \
    && mkdir -p /tmp/deps \
    && cd /tmp/deps \
    && curl -L https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -rf Miniconda3-latest-Linux-x86_64.sh \
    && PATH=/opt/miniconda3/bin:$PATH \
    && conda install conda=4.5.11 python=3.5 \
    && conda install pip wheel opencv \
    && conda clean -y --all \
    && cd / \
    && rm -rf /tmp/*

ENV PATH /opt/miniconda3/bin:$PATH

RUN pip install nnabla
