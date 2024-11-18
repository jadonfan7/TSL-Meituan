#!/bin/bash

#SBATCH --job-name=TSL
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a800:2
#SBATCH --mail-type=end
#SBATCH --mail-user=2151177@tongji.edu.cn
#SBATCH --output=%j.out 
#SBATCH --error=%j.err


module load cuda/11.8

srun python /share/home/tj23028/TSL/IPPO/train/train.py