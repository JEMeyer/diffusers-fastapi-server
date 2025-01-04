FROM --platform=$BUILDPLATFORM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS amd64_base
FROM --platform=$BUILDPLATFORM arm64v8/python:3.9-slim AS arm64_base

# Select the appropriate base image
FROM amd64_base AS base
COPY --from=arm64_base / /

# Install additional dependencies
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the current project folder to /app
COPY . .

# Install the diffusors_fastapi_server package and its dependencies
RUN pip3 install . --no-cache-dir

# Expose the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "diffusors_fastapi_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
