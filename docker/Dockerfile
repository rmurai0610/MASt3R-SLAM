FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS build

# ARG TORCH_CUDA_ARCH_LIST=8.9
# ARG CUDA_HOME
# ARG PATH
# ARG LD_LIBRARY_PATH

# ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
# ENV CUDA_HOME=${CUDA_HOME}
# ENV PATH=${PATH}
# ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

# ENV TORCH_CUDA_ARCH_LIST=8.9
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=/usr/local/cuda/bin:$PATH
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  curl \
  python3.11 \
  python3.11-dev \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python

RUN pip install --upgrade pip && \
  pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
COPY ./ /workspace

RUN pip install -e thirdparty/mast3r
RUN pip install -e thirdparty/in3d
RUN pip install --no-build-isolation -e .

FROM build AS deploy

RUN apt-get purge -y \
  build-essential \
  cmake \
  git \
  curl \
  && apt-get autoremove -y

RUN apt-get update && apt-get install -y \
  libgl1-mesa-dev \
  libgl1-mesa-glx \
  libegl1-mesa \
  libegl1-mesa-dev \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

CMD ["bash"]
