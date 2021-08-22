#!/bin/bash
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-12:0:0
#SBATCH --account=def-uofavis-ab

nvidia-smi

module load python/3.8
source ~/ENV_new/bin/activate

export MPLBACKEND=agg

python -m research.dmri_hippo.run augmentation_experiment_grid \
~/projects/def-uofavis-ab/shared_data/Diffusion_MRI_cropped.tar \
~/scratch/Checkpoints/ \
--work_path ${SLURM_TMPDIR}/${SLURM_TASK_ID} \
--task_id $SLURM_ARRAY_TASK_ID
