# Use CUDA 12.2 runtime as base image
# This version is compatible with PyTorch 2.5.0 which requires CUDA >= 12.1
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    python-is-python3 \
    python3-pip \
    python3.10-dev \
    git \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Create and switch to a new user
RUN adduser --disabled-password --gecos '' -u 1000 user
RUN mkdir -p /opt/ml/code && chown -R user:user /opt/ml/code
WORKDIR /opt/ml/code

# Install PyTorch with CUDA 12.1 support
RUN pip3 install --upgrade pip && \
    pip3 install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

# Copy files and set permissions
COPY --chown=user:user example.py serve.sh /opt/ml/code/
RUN chmod +x /opt/ml/code/serve.sh

# Switch to non-root user
USER user

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD ["/opt/ml/code/serve.sh"]
