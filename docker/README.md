## Dockerized Deep QA 


Build *deepqa:latest* image:

```sh
docker build -t deepqa:latest .
```

Run the `data-dirs.sh` script and copy your models and data into the appropriate folders:

```sh
cd DeepQA/docker
./data-dirs.sh
```

Start the server with docker-compose:

```sh
docker-compose -f <YAML file> up
```

TODO: Nvidia-Docker
