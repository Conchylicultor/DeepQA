## Dockerized Deep QA

### Instructions

**Step 1**: Build *deepqa:latest* image, from the root directory:

```sh
docker build -t deepqa:latest .
# Or the GPU version:
docker build -t deepqa:latest -f Dockerfile.gpu .
```

**Step 2**: Run the `data-dirs.sh` script and indicate the folder were you want the docker data to be stored (model files, dataset, logs):

```sh
cd DeepQA/docker
./data_dirs.sh <base_dir>
```

Warning: The `data/` folder will be entirely copied from `DeepQA/data`. If you're not running the script from a fresh clone and have downloaded a big dataset, this can take a while.

**Step 3**: Copy the model you want (ex: the pre-trained model) inside `<base_dir>/model-server`.

**Step 4**: Start the server with docker-compose:

```sh
DEEPQA_WORKDIR=<base_dir> docker-compose -f deploy.yml up
# Or the GPU version:
DEEPQA_WORKDIR=<base_dir> nvidia-docker-compose -f deploy.yml up
```

After the server is launched, you should be able to speak with the ChatBot at [http://localhost:8000/](http://localhost:8000/).

**Note**: You can also train a model with the previous command by replacing `deploy.yml` by `train.yml`.

### For the GPU version

Note that the GPU version require [Nvidia-Docker](https://github.com/NVIDIA/nvidia-docker). In addition, to install `nvidia-docker-compose`:

```sh
pip install jinja2
pip install nvidia-docker-compose
```
