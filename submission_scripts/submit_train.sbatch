#!/bin/bash
#SBATCH -J pwf_train
#SBATCH -p GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --mem=96000

## Load my profile and pytorch info
source ~/.bash_profile
ml cuda/11.1 python/intel21
conda deactivate
conda activate thepayne

cd /home/aji/train_payne_rpa3/relu300
pwf_train relu300.cfg
