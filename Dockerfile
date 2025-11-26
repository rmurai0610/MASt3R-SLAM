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
  libglib2.0-0 \
  libgl1-mesa-glx \
  libgomp1 \
  libgstreamer1.0-0 \
  libgstreamer-plugins-base1.0-0 \
  libgtk-3-0 \
  ffmpeg \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libv4l-dev \
  libxvidcore-dev \
  libx264-dev \
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

# Install PyTorch via pip (more reliable than conda in Docker due to fewer library dependencies)
# This avoids the Intel VTune library linking issues that occur with conda pytorch
RUN conda run -n mast3r-slam pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Clone the repository
RUN git clone --recursive https://github.com/rmurai0610/MASt3R-SLAM.git /app
WORKDIR /app

# Set environment variables for compiling CUDA extensions
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV CUDA_HOME=/usr/local/cuda

# Run all subsequent commands inside a single RUN layer with the conda environment explicitly activated.
# This ensures all pip subprocesses correctly inherit the environment and can find PyTorch.
# Using SHELL to set bash with conda init, so all RUN commands use the activated environment.
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
  conda activate mast3r-slam && \
  pip install \
  numpy==1.26.4 \
  einops \
  pyrealsense2 \
  evo \
  natsort \
  plyfile \
  setuptools==70.0.0 \
  torchcodec==0.1 \
  opencv-python==4.10.0.84 \
  opencv-contrib-python==4.10.0.84 && \
  pip install --no-build-isolation -e thirdparty/mast3r && \
  pip install -e thirdparty/in3d && \
  sed -i 's/has_cuda = torch.cuda.is_available()/has_cuda = True/' setup.py && \
  pip install --no-build-isolation --no-binary lietorch -e .

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

# Fix multiprocessing shared memory issue in Docker
RUN echo 'none /dev/shm tmpfs rw,nosuid,nodev,noexec,relatime,size=2g 0 0' >> /etc/fstab

# Add conda activation to bashrc
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate mast3r-slam" >> /root/.bashrc

# Set the final entrypoint to launch a bash shell within the conda environment
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]