FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables for non-interactive installs and Python
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential git wget curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Copy your application code to the container
WORKDIR /app
COPY . /app

# Install Python dependencies from setup.py
RUN pip install .

# Ensure the NVIDIA runtime is set up correctly
RUN pip install nvidia-pyindex && pip install nvidia-torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Expose the port used by the server
EXPOSE 8000

# Run the server
CMD ["uvicorn", "diffusors_fastapi_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
