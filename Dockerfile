# Base image for amd64 with CUDA support
FROM --platform=$BUILDPLATFORM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS amd64_base

# Base image for arm64 with Python 3.10 slim
FROM --platform=$BUILDPLATFORM arm64v8/python:3.10-slim AS arm64_base

# Select the appropriate base image based on the platform
FROM amd64_base AS base
COPY --from=arm64_base / /

# Set environment variables to optimize Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current project folder to /app
COPY . .

# Install the application and its dependencies
RUN pip3 install --upgrade pip && pip3 install . --no-cache-dir

# Expose the port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "diffusors_fastapi_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
