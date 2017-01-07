## Dockerized Deep QA


Build *deepqa:latest* image, from the root directory:

```sh
docker build -t deepqa:latest .
# Or the GPU version:
docker build -t deepqa:latest -f Dockerfile.gpu .
```

Run the `data-dirs.sh` script and copy your models and data into the appropriate folders:

```sh
cd DeepQA/docker
./data-dirs.sh
```

Start the server with docker-compose:

```sh
docker-compose -f <YAML file> up
# Or the GPU version:
nvidia-docker-compose -f <YAML file> up
```

Note that the GPU version require [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker). In addition, to install `nvidia-docker-compose`:

```sh
pip install jinja2
pip install nvidia-docker-compose
```
