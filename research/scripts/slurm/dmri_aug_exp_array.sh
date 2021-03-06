#!/bin/bash
#SBATCH --nodes=1
#SBATCH --array=0-6
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=0-8:0:0
#SBATCH --account=def-uofavis-ab

nvidia-smi

module load python/3.8
source ~/ENV_new/bin/activate

export MPLBACKEND=agg

python research/dmri_hippo/generate_parallel_commands.py | parallel --jobs 3