# Use NVIDIA CUDA 12.1.1 development environment as the base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install system packages required for the build (git, wget, ninja, etc.)
RUN apt-get update && apt-get install -y \
  git \
  wget \
  build-essential \
  ninja-build \
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

# Set the shell to use the 'mast3r-slam' conda environment by default
SHELL ["conda", "run", "-n", "mast3r-slam", "/bin/bash", "-c"]

# Install the exact PyTorch versions from the README
RUN conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Clone the repository
RUN git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git /app
WORKDIR /app

# Install all remaining dependencies using pip within the conda environment
RUN pip install \
  numpy==1.26.4 \
  einops \
  pyrealsense2 \
  evo \
  natsort \
  plyfile \
  setuptools==70.0.0 \
  torchcodec==0.1

# Set environment variables for compiling CUDA extensions
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda

# Install the submodules and the main project in editable mode
RUN pip install -e thirdparty/mast3r
RUN pip install -e thirdparty/in3d

# --- THE FINAL FIX: Patch the main setup.py to force CUDA compilation ---
RUN sed -i 's/has_cuda = torch.cuda.is_available()/has_cuda = True/' setup.py

# Finally, install the main project
RUN pip install --no-build-isolation -e .

# Set the final entrypoint to launch a bash shell within the conda environment
CMD ["conda", "run", "-n", "mast3r-slam", "bash"]