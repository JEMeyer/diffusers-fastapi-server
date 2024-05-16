FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Set the working directory
WORKDIR /app

# Copy the current project folder to /app
COPY . .

# Install the stablediffusion_fastapi_multigpu package
RUN pip3 install .

# Export the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "stablediffusion_fastapi_multigpu.app:app", "--host", "0.0.0.0", "--port", "8000"]
