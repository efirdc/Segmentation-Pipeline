#!/bin/bash

module load python/3.8
source ~/ENV_new/bin/activate

TASK_NAME="Task501_DMRI_Hippocampus_Whole"
FOLDER_WITH_TEST_CASES="${nnUNet_raw_data_base}/nnUNet_raw_data/${TASK_NAME}/imagesTs/"

OUTPUT_NAME="predictionsTs"
OUTPUT_ROOT="${RESULTS_FOLDER}/nnUNet/inference/${TASK_NAME}/${OUTPUT_NAME}/"

OUTPUT_FOLDER_MODEL1="${OUTPUT_ROOT}/2d/"
OUTPUT_FOLDER_MODEL2="${OUTPUT_ROOT}/3d_fullres/"
OUTPUT_FOLDER="${OUTPUT_ROOT}/ensemble/"

ENSEMBLE_FOLDER="ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1"

nnUNet_predict \
  -i $FOLDER_WITH_TEST_CASES \
  -o $OUTPUT_FOLDER_MODEL1 \
  -tr nnUNetTrainerV2 \
  -ctr nnUNetTrainerV2CascadeFullRes \
  -m 2d \
  -p nnUNetPlansv2.1 \
  -t $TASK_NAME \
  --save_npz
nnUNet_predict \
  -i $FOLDER_WITH_TEST_CASES \
  -o $OUTPUT_FOLDER_MODEL2 \
  -tr nnUNetTrainerV2 \
  -ctr nnUNetTrainerV2CascadeFullRes \
  -m 3d_fullres \
  -p nnUNetPlansv2.1 \
  -t $TASK_NAME \
  --save_npz
nnUNet_ensemble \
  -f $OUTPUT_FOLDER_MODEL1 $OUTPUT_FOLDER_MODEL2 \
  -o $OUTPUT_FOLDER \
  -pp "${RESULTS_FOLDER}/nnUNet/ensembles/${TASK_NAME}/${ENSEMBLE_FOLDER}/postprocessing.json" \
  --npz