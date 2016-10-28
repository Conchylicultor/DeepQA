## Dockerfile to build DeepQ&A container image

FROM python:3.5.2
MAINTAINER vt

## dependencies
RUN \
  apt-get -qq -y update && apt-get -y install unzip
  
RUN  \
  pip3 install -U nltk \
  tqdm \
  django \
  asgi_redis \
  channels && \
  python3 -m nltk.downloader punkt

## Tensorflow  
ARG TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl
  
RUN \
  pip3 install -U $TF_BINARY_URL

COPY ./ /root/DeepQA


## Run Config
ENV CHATBOT_SECRET_KEY="my-secret-key"
EXPOSE 8000
COPY docker/settings.py /root/DeepQA/chatbot_website/chatbot_website/
COPY docker/chatbot.sh /root/DeepQA/chatbot_website/
WORKDIR /root/DeepQA/chatbot_website
CMD bash chatbot.sh
