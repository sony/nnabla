FROM nnabla/nnabla-ext-cuda-multi-gpu:py38-cuda100-multi-gpu-ubuntu18-v1.8.0

LABEL maintainer="Kauki.Yoshiyama@sony.com"

RUN apt-get update
RUN apt-get install -y \
  less \
  libgl1-mesa-glx \
  libgomp1

ENV HTTP_PROXY ${http_proxy}

RUN pip install --proxy ${HTTP_PROXY} open3d scikit-image scipy tqdm


