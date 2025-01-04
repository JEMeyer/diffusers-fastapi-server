# Diffusors FastAPI Server

This project is a web server that provides a FastAPI-based interface for generating images using Diffusors models. It supports both text-to-image (`/txt2img`) and image-to-image (`/img2img`) generation endpoints. The server is designed to only run on 1 gpu, if multiple are needed, use ai-maestro-router in conjunction with multiple `diffusors-fastapi-server` instances

## Prerequisites

- Docker
- NVIDIA GPU with driver installed
- NVIDIA Container Toolkit

## Usage

### Docker Run

The container accepts an environment variable `MODEL_NAME` which will default to `stabilityai/sdxl-turbo`. You can swap out this with any other model that works with the Diffusion pipelines AutoPipelineForText2Image and AutoPipelineForImage2Image.

In order to make /img2img and /upload work, redis must be setup. Pass in `REDIS_HOST` and `REDIS_PORT` (defaults are `localhost` and `6379`).

Ports and gpus are configured with standard docker flags.

To use the most recent stable image, pull the `latest` tag:

```bash
docker run -e MODEL_NAME=stabilityai/sdxl-turbo  -p 8000:8000 ghcr.io/jemeyer/diffusors-fastapi-server:latest
```

This will start the server and make it accessible at <http://localhost:8000>.

#### GPU Configuration

If you have an NVIDIA GPU and want to use it with the UI, you can pass the --gpus flag to docker run:

- To use all available GPUs (only 1 - if multiple are needed, use ai-maestro-router):

```bash
docker run --gpus all -e MODEL_NAME=stabilityai/sdxl-turbo -p 8000:8000 ghcr.io/jemeyer/diffusors-fastapi-server:latest
```

- To use a specific number of GPUs:

```bash
docker run --gpus 1 -e MODEL_NAME=stabilityai/sdxl-turbo -p 8000:8000 ghcr.io/jemeyer/diffusors-fastapi-server:latest
```

- To use a specific GPU by its device ID (e.g., GPU 2), and define a different model_name:

```bash
docker run --gpus -e MODEL_NAME=stabilityai/sd-turbo device=2 -p 8000:8000 ghcr.io/jemeyer/diffusors-fastapi-server:latest
```

Note that you need to have the NVIDIA Container Toolkit installed on your host for GPU passthrough to work.

### Docker Compose

You can also use Diffusors Fastapi Server with Docker Compose. Here's an example docker-compose.yml file:

```yaml
services:
  diffusors-fastapi-server:
    image: ghcr.io/jemeyer/diffusors-fastapi-server:latest
    ports:
      - 8000:8000
    environment:
      - MODEL_NAME=stabilityai/sdxl-turbo
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

This configuration will start a container using the latest image and make it accessible at <http://localhost:8000>. It also configures the container to use 1 GPU.

To use a specific GPU, you can use the device_ids property instead of count:

```yaml
reservations:
  devices:
    - driver: nvidia
      device_ids: ["2"]
      capabilities: [gpu]
```

Start the container with:

```bash
docker-compose up -d
```

### Endpoints

- `/txt2img`: Generate an image from a text prompt
  - `curl -H "Content-Type: application/json" -d '{"prompt": "PROMPT"}' http://IP_ADDR:8000/txt2img --output t2i_generated_image.png`
  - Returns a fastAPI StreamingResponse of type `image/png`
- `/upload`: Upload an image (to then be refereneced by `img2img`)
  - `curl -X POST -F "file=@./t2i_generated_image.jpg" http://IP_ADDR:8000/upload`
  - Returns `{"file_id":"FILE_ID"}`
- `/img2img`: Generate an image based on an input image id and a text prompt
  - `curl -H "Content-Type: application/json" -d '{"file_id":"FILE_ID", "prompt": "PROMPT"}' http://IP_ADDR:8000/img2img --output i2igenerated_image.png`
  - Returns a fastAPI StreamingResponse of type `image/png`

This server uses `-turbo` SD models, and as such no negPrompt is used.

## Hardware

Below you will find a comparison table that outlines the specifications of various GPUs, the model used, the amount of VRAM required for operation, and the observed range for image generation speed. It's important to note that these values are not hard limits but rather observations based on running these models under certain conditions. As such, your results may vary.

| GPU                     | Model Used | VRAM Used | Generation Speed Range |
| ----------------------- | ---------- | --------- | ---------------------- |
| NVIDIA GeForce RTX 3090 | SDXL-Turbo | 10.8 GiB  | .65 - .70 seconds      |
| NVIDIA GeForce RTX 3090 | SD-Turbo   | 4.25 GiB  | .42 - .50 seconds      |
| NVIDIA Tesla P100       | SDXL-Turbo | 10 GiB    | 1.8 - 2 seconds        |
| NVIDIA Tesla P100       | SD-Turbo   | 4.7 GiB   | 1.3 - 1.5 seconds      |

## Development

1. Clone the repository:

   ```bash
   git clone https://github.com/jemeyer/diffusors-fastapi-server.git
   cd diffusors-fastapi-server
   ```

2. Set up a virtual environment using `pyenv`:

   ```bash
   pyenv virtualenv 3.10.13 diffusors-fastapi-server
   pyenv local diffusors-fastapi-server
   ```

3. Install the required dependencies:

   ```bash
   pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/
   pip install .
   ```

   Note: The `torch` package is explicitly installed from the PyTorch download URL to ensure compatibility with the CUDA version.

## Dockerization

The server can be dockerized for easy deployment and scalability. A Dockerfile is provided to build the Docker image.

To build the Docker image, run the following command:

```bash
docker build -t jemeyer/diffusors-fastapi-server .
```

To run the Docker container, use the following command:

```bash
docker run -p 8000:8000 jemeyer/diffusors-fastapi-server
```

The server will be accessible at `http://localhost:8000`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under Apache 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Stable Diffusion](https://stability.ai/stable-image)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
