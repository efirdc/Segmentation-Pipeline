#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gres=gpu:v100l:2
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=200M
#SBATCH --time=0-0:2:0
#SBATCH --account=def-uofavis-ab

nvidia-smi
printenv

srun nvidia-smi
srun printenv

