#!/bin/bash

mkdir -p ~/deepQA
cd ~/deepQA

mkdir -p logs
mkdir -p data/samples
mkdir -p data/cornell  # Not necessary if samples are presents
mkdir -p save/model-server
ln -s save/model-server model-server
