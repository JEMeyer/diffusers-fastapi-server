# Use the official PyTorch image with CUDA 12.1
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

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

# Copy and install Python dependencies (excluding torch, torchvision, torchaudio)
COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Save the updated requirements to a file for logging/reference
RUN /opt/venv/bin/pip freeze > updated_requirements.txt

# Copy application code
COPY app.py .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
