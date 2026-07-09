# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG AITK_VERSION=1.5.0
ARG GIT_COMMIT=unknown

LABEL authors="jaret" \
    org.opencontainers.image.title="AITK Studio" \
    org.opencontainers.image.description="AITK Studio UI and diffusion-model training toolkit." \
    org.opencontainers.image.source="https://github.com/BadAtCaptchas/AITK-Studio" \
    org.opencontainers.image.version="${AITK_VERSION}" \
    org.opencontainers.image.revision="${GIT_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
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

WORKDIR /tmp
RUN curl -fsSL https://deb.nodesource.com/setup_22.x -o nodesource_setup.sh \
    && bash nodesource_setup.sh \
    && apt-get update \
    && apt-get install --no-install-recommends -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/nodesource_setup.sh

# Optional managed Cloudflare Tunnel and local Ollama support.
RUN curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb \
    && apt-get update \
    && apt-get install --no-install-recommends -y /tmp/cloudflared.deb \
    && rm -f /tmp/cloudflared.deb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app/ai-toolkit

# Install large dependencies before copying frequently changing application code.
RUN python -m pip install --no-cache-dir --break-system-packages \
      torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchcodec==0.10.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple

COPY requirements_base.txt requirements.txt ./
RUN python -m pip install --no-cache-dir --break-system-packages -r requirements.txt \
    && python -m pip install --no-cache-dir --break-system-packages setuptools==70.0.0

COPY ui/package.json ui/package-lock.json ./ui/
WORKDIR /app/ai-toolkit/ui
RUN npm ci

# Build exactly the source supplied as the Docker build context. This avoids a
# mutable network clone and ensures the OCI revision label describes the code.
WORKDIR /app/ai-toolkit
COPY . .

WORKDIR /app/ai-toolkit/ui
RUN npm run update_db \
    && npm run build \
    && rm -f /app/ai-toolkit/aitk_db.db /app/ai-toolkit/aitk_db.db-shm /app/ai-toolkit/aitk_db.db-wal \
    && npm cache clean --force

EXPOSE 8675 6006

WORKDIR /
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
