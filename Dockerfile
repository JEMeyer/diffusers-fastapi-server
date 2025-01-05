# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables for non-interactive installs and Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PATH="/opt/venv/bin:$PATH"

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    wget curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a Python virtual environment
RUN python3.10 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Application stage
FROM base AS app

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install application dependencies
RUN pip install . \
    && pip install nvidia-pyindex nvidia-torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Expose the port used by the server
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "diffusors_fastapi_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
