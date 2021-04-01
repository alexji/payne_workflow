#!/bin/bash
#SBATCH -J pwf_fitgrid
#SBATCH -p OBS
#SBATCH --time=4:00:00
#SBATCH --array=0-326

## Load my profile and pytorch info
source ~/.bash_profile
ml cuda/11.1 python/intel21
conda deactivate
conda activate thepayne

cd /home/aji/train_payne_rpa3/relu300
pwf_fitgrid relu300.cfg ${SLURM_ARRAY_TASK_ID} 327 100
