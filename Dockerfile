# Sử dụng image PyTorch với CUDA 12.4 và cuDNN 9
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Đặt CUDA home environment variable
ENV CUDA_HOME /usr/local/cuda

# Cài đặt các công cụ cần thiết và làm sạch cache sau khi cài đặt
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    unzip \
    stow \
    subversion \
    fasd \
    libfreetype6-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật pip và setuptools
RUN pip install --upgrade pip setuptools

RUN pip install --no-cache-dir \
    sagemaker-training \
    huggingface-hub \
    transformers \
    peft \
    Pillow \
    numpy \
    einops \
    tqdm \
    dataclasses \
    torchvision 


RUN pip install packaging ninja
RUN pip install flash-attn==v2.1.1 --no-build-isolation
RUN pip install git+https://github.com/HazyResearch/flash-attention.git@v2.1.1#subdirectory=csrc/rotary

# Sao chép mã nguồn từ thư mục `src` của bạn vào thư mục `/opt/ml/code` trong container
COPY ./src /opt/ml/code

# Đặt biến môi trường cho chương trình SageMaker sẽ chạy
ENV SAGEMAKER_PROGRAM train.py

# Đặt thư mục làm việc
# WORKDIR /opt/ml/code
