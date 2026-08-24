FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
ENV HUGGINGFACE_HUB_CACHE=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/huggingface

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY handler.py .
COPY prompt.json .
COPY details.json .

CMD ["python3", "-u", "handler.py"]
