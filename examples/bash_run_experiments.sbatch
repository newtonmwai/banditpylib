#!/usr/bin/bash

#SBATCH -t 1-00:00:00
#SBATCH -N 1 --gpus-per-node=T4:1
#SBATCH -p alvis

if [ -z "$rho" ]
then
    rho=0.5
fi

if [ -z "$batch_size" ]
then
    batch_size=50
fi

if [ -z "$means" ]
then
    means='m1'
fi

. load_modules.sh

workon mwai_env_bundle & python experiments.py -r $rho -b $batch_size -m $means