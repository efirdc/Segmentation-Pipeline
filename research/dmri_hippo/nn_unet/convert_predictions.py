from pathlib import Path
import json
import shutil
import os
import copy

import torch
import torchio as tio
import numpy as np

from segmentation_pipeline import *
from research.dmri_hippo.configs.main_config import get_context


def main(
        task_name: str,
        task_id: int,
        out_path: str,
        split: bool,
        dataset_path: str
):
    cv_root = Path(f"{os.environ['RESULTS_FOLDER']}/nnUNet/ensembles/{task_name}")
    long_name = "ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1"

    cv_nifti_path = cv_root / long_name / 'ensembled_postprocessed'
    test_nifti_path = Path(f"{os.environ['RESULTS_FOLDER']}/nnUNet/inference/{task_name}/predictionsTs/ensemble")

    nifti_file_paths = [
        path for path in list(cv_nifti_path.iterdir()) + list(test_nifti_path.iterdir())
        if path.suffix == ".gz"
    ]

    subject_names_path = Path(f"X:/Datasets/nnUNet_raw_data_base/nnUNet_raw_data/{task_name}/original_subject_names.json")
    with subject_names_path.open() as f:
        names = json.load(f)
        cv_names = {v: k for k, v in names['cross_validation_subjects'].items()}
        test_names = {v: k for k, v in names['test_subjects'].items()}
        original_name_lookup = {**cv_names, **test_names}

    out_path = Path(out_path)

    if not split:
        for file_path in nifti_file_paths:
            name = file_path.stem.split(".")[0]
            original_name = original_name_lookup[name]

            out_dir = out_path / original_name
            out_dir.mkdir(exist_ok=True)

            shutil.copy(file_path, out_dir / f"whole_roi_pred_task{task_id}.nii.gz")
    else:
        context = get_context(device=torch.device("cuda"),
                              variables={"DATASET_PATH": dataset_path})
        context.init_components()

        dataset: SubjectFolder = context.dataset
        dataset.set_transform(tio.CropOrPad((96, 88, 20), mask_name='whole_roi_union'))
        sample_subject: tio.Subject = dataset[0]
        sample_transform = sample_subject.get_composed_history()

        inverse_transform = tio.Compose([
            CustomRemapLabels(remapping={1: 2}, masking_method="Right"),
            sample_transform.inverse(warn=False),
        ])

        nifti_file_paths.sort(key=lambda p: int(p.name.split(".")[0].split("_")[1]))
        nifti_file_path_pairs = zip(nifti_file_paths[::2], nifti_file_paths[1::2])

        for left_file_path, right_file_path in nifti_file_path_pairs:

            left_name = left_file_path.stem.split(".")[0]
            right_name = right_file_path.stem.split(".")[0]
            left_original_name = original_name_lookup[left_name]
            right_original_name = original_name_lookup[right_name]

            original_name = "_".join(left_original_name.split("_")[:-1])

            left_label_map = tio.LabelMap(left_file_path)
            right_label_map = tio.LabelMap(right_file_path)
            left_label_map.load()
            right_label_map.load()

            right_label_map = tio.Flip(axes=(0,))(right_label_map)
            right_label_map = tio.Pad(padding=(0, 48, 0, 0, 0, 0))(right_label_map)
            left_label_map = tio.Pad(padding=(48, 0, 0, 0, 0, 0))(left_label_map)

            combined_tensor = right_label_map.data + left_label_map.data

            subject = dataset.all_subjects_map[original_name]
            affine = subject['mean_dwi'].affine

            label_map = tio.LabelMap(tensor=combined_tensor, affine=affine)
            label_map = inverse_transform(label_map)

            out_dir = out_path / "subjects" / original_name
            out_dir.mkdir(exist_ok=True, parents=True)
            out_file = out_dir / f"whole_roi_pred_task{task_id}.nii.gz"
            label_map.save(out_file)
            print("Saved", out_file)


if __name__ == "__main__":
    main(
        task_name="Task502_DMRI_Hippocampus_Whole_Split",
        task_id=502,
        out_path="X:/Predictions/Diffusion_MRI/nnUNet/",
        split=True,
        dataset_path="X:/Datasets/Diffusion_MRI"
    )
