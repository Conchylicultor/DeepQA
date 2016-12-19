## Dockerized Deep QA 


Build *deepqa:latest* image:

```sh
docker build -t deepqa:latest .
```

Run the `data-dirs.sh` script and copy your models and data into the appropriate folders:

```sh
cd DeepQA/docker
./data-dirs.sh <base_dir>
```

Start the server with docker-compose:

```sh
DEEPQA_WORKDIR=<base_dir> docker-compose -f <YAML file> up
```

TODO: Nvidia-Docker
