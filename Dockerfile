# Use NVIDIA's CUDA runtime with Ubuntu
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    build-essential git curl libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate Python virtual environment
RUN python3.10 -m venv /opt/venv
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 separately
# Ensure 'torch' is removed from requirements.txt to avoid duplication
RUN /opt/venv/bin/pip install torch==2.1.0+cu121 torchvision==0.15.2+cu121 torchaudio==2.0.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy and install Python dependencies
COPY requirements.txt .
RUN /opt/venv/bin/pip install --upgrade --no-cache-dir -r requirements.txt

# Save the updated requirements to a file for logging/reference
RUN /opt/venv/bin/pip freeze > updated_requirements.txt

# Copy application code
COPY app.py .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
