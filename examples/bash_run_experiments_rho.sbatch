#!/usr/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis

if [ -z "$rho" ]
then
    epsilon=0.1
fi

. load_modules.sh

call workon mwai_env_bundle & python experiments.py -r $rho
