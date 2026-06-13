FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

LABEL authors="jaret"

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# Include Blackwell sm_120 for RTX 50-series and DGX Spark class GPUs.
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    tmux \
    htop \
    nvtop \
    python3-opencv \
    openssh-client \
    openssh-server \
    openssl \
    rsync \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install nodejs
WORKDIR /tmp
RUN curl -sL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cloudflared for optional managed Cloudflare Tunnel support.
RUN curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb && \
    apt-get update && \
    apt-get install -y /tmp/cloudflared.deb && \
    rm -f /tmp/cloudflared.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama for optional secure remote caption workers. The startup script
# keeps it bound to localhost unless explicitly overridden.
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Set aliases for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# install pytorch before cache bust to avoid redownloading pytorch
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchcodec==0.10.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --break-system-packages

# Fix cache busting by moving CACHEBUST to right before git clone
ARG CACHEBUST=1234
ARG GIT_COMMIT=main
RUN echo "Cache bust: ${CACHEBUST}" && \
    git clone https://github.com/BadAtCaptchas/ai-toolkit.git && \
    cd ai-toolkit && \
    git checkout ${GIT_COMMIT}

WORKDIR /app/ai-toolkit

# copy requirements.txt to leverage docker cache
COPY requirements.txt /app/ai-toolkit/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip install setuptools==70.0.0 --no-cache-dir --break-system-packages

# Build UI
WORKDIR /app/ai-toolkit/ui
RUN npm install && \
    npm run update_db && \
    npm run build

# Expose UI and optional TensorBoard ports.
EXPOSE 8675 6006

WORKDIR /

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
