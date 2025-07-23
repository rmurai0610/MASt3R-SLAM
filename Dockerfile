# Use the stable NVIDIA CUDA image for Ubuntu 22.04
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install all necessary system dependencies, including those for OpenCV
RUN apt-get update && apt-get install -y \
  git \
  wget \
  build-essential \
  ninja-build \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libglib2.0-0 \
  libgtk-3-0 \
  libgtk-3-dev \
  libusb-1.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
  /bin/bash ~/miniconda.sh -b -p /opt/conda && \
  rm ~/miniconda.sh

# Add conda to the system's PATH environment variable
ENV PATH /opt/conda/bin:$PATH

# Accept the Anaconda Terms of Service
RUN conda config --set auto_activate_base false && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the conda environment with Python 3.11
RUN conda create -n mast3r-slam python=3.11 -y

# Use the exact command from the README to install PyTorch within the new environment
RUN conda run -n mast3r-slam conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Clone the repository
RUN git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git /app
WORKDIR /app

# Set environment variables for compiling CUDA extensions
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda

# Run all subsequent commands inside a single RUN layer with the conda environment explicitly activated.
# This ensures all pip subprocesses correctly inherit the environment and can find PyTorch.
RUN conda run -n mast3r-slam /bin/bash -c " \
  pip install \
  numpy==1.26.4 \
  einops \
  pyrealsense2 \
  evo \
  natsort \
  plyfile \
  setuptools==70.0.0 \
  torchcodec==0.1 \
  opencv-python && \
  pip install -e thirdparty/mast3r && \
  pip install -e thirdparty/in3d && \
  sed -i 's/has_cuda = torch.cuda.is_available()/has_cuda = True/' setup.py && \
  pip install --no-build-isolation --no-binary lietorch -e . \
  "

# Create a shell script that activates conda environment
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate mast3r-slam\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Also create a wrapper for python to always use the conda env
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate mast3r-slam\n\
python "$@"' > /usr/local/bin/python-mast3r && \
    chmod +x /usr/local/bin/python-mast3r

# Add conda activation to bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate mast3r-slam" >> /root/.bashrc

# Set the final entrypoint to launch a bash shell within the conda environment
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]