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

if [[ "$(uname -s)"  == "Darwin" ]]; then
    if [[ -z $(command -v greadlink) ]]; then
        read -r -p 'no greadlink, install with brew? y/N: ' response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
        then
            brew install coreutils
        else
            echo -e "can't continue without proper readlink!\n"
            exit 1
        fi
    else
    gitdir=$(greadlink -f .)
    fi
else
    gitdir=$(readlink -f .)
fi

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
