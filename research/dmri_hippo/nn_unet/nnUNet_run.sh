#!/bin/bash
#SBATCH --array=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=0-3:0:0
#SBATCH --account=def-uofavis-ab

TASK_NAME=Task502_DMRI_Hippocampus_Whole_Split

echo r$SLURM_ARRAY_TASK_ID $TASK_NAME job start

nvidia-smi

module load python/3.8
source ~/ENV_new/bin/activate

echo r$SLURM_ARRAY_TASK_ID copying dataset to SSD...
mkdir -p $SLURM_TMPDIR/nnUNet_preprocessed/$TASK_NAME/
cp -r /scratch/efirdc/nnUNet_preprocessed/$TASK_NAME/ $SLURM_TMPDIR/nnUNet_preprocessed/
echo r$SLURM_ARRAY_TASK_ID copy finished

export nnUNet_preprocessed=$SLURM_TMPDIR/nnUNet_preprocessed/
export nnUNet_raw_data_base="/scratch/efirdc/nnUNet_raw_data_base/"
export RESULTS_FOLDER="/scratch/efirdc/nnUNet_trained_models/"

export nnUNet_n_proc_DA=$SLURM_CPUS_PER_TASK

echo r$SLURM_ARRAY_TASK_ID starting training

# Notes:
# Add -c flag to resume a run, but make sure to remove it if not resuming else it will crash
# First param is the model type. Train '2d' and '3d_fullres' for the dmri data
nnUNet_train 2d nnUNetTrainerV2 $TASK_NAME $SLURM_ARRAY_TASK_ID -c --npz