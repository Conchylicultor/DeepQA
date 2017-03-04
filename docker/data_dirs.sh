#!/bin/bash

###
#
# Should be run from DeepQA/docker
#
# $1: work directory where to create deepQA's data
#
###

workdir="$1"
workdir=${workdir:="${DEEPQA_WORKDIR}"}
gitdir=$(readlink -f .)

echo "Creating:"
echo " - ${workdir}"
echo "From:"
echo " - ${gitdir}"

mkdir -p ${workdir}
cd ${workdir}

mkdir -p logs
cp -r ${gitdir}/../data ${workdir}
mkdir -p save/model-server
ln -s save/model-server model-server
