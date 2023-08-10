#!/usr/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis

if [ -z "$batch_size" ]
then
    batch_size=20
fi

. load_modules.sh

call workon mwai_env_bundle & python experiments.py -b $batch_size
