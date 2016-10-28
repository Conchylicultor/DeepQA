## Dockerized Deep QA 

To start the server with models and samples run the `data-dir.sh` script and move them to the appropriate folder:

```sh
cd DeepQA/docker
./data-dir.sh
```

must build *deepqa:latest* image first:

```sh
docker build -t deepqa:latest .
```

start server with docker-compose:

```sh
docker-compose -f <YAML file> up
```



