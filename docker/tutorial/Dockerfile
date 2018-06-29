FROM ubuntu:16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-setuptools python3-wheel git\
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install nnabla
RUN pip3 install jupyter
RUN pip3 install sklearn
RUN pip3 install imageio

RUN git clone --depth=1 https://github.com/sony/nnabla && mv nnabla/tutorial . && rm -rf nnabla
RUN git clone --depth=1 https://github.com/sony/nnabla-examples && rm -rf nnabla-examples/.git  && mv nnabla-examples tutorial

WORKDIR /tutorial

CMD ["jupyter", "notebook", "--ip=*", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token='nnabla'"]
