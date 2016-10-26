## Dockerized Deep QA 

To start the server with models and samples run the `data-dir.sh` script and move them to the appropriate folder:

```sh
cd deepQAdk
./data-dir.sh
cp ~/deepQA/samples 
unzip -d ~/deepQA/model model.zip 
```

start server with docker-compose:
```sh
docker-compose up
```

then go to your browser and enter `localhost:8000` or `127.0.0.1:8000`

logs are written to `~/deepQA/logs`

### Note:
- must build *af.cds.bns:5001/cmo/deepqa* image first since compose build doesn't work behind the proxy yet:

```sh
docker build -t af.cds.bns:5001/cmo/deepqa .
# must be logged in to push to the repo
docker push af.cds.bns:5001/cmo/deepqa
```

