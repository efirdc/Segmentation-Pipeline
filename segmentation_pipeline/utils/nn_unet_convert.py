import os
import json
import copy
from typing import Sequence
from pathlib import Path
from collections import OrderedDict
import pickle
from typing import Optional

import numpy as np

# TODO: Figure out why importing SubjectFolder is a circular import
# from ..data_processing import SubjectFolder
from ..transforms import CustomSequentialLabels


def save_dataset_as_nn_unet(
        cross_validation_dataset,
        output_path: str,
        short_name: str,
        image_names: Sequence[str],
        label_map_name: str,
        test_dataset: Optional = None,
        metadata: dict = None,
        output_folds: bool = False,
        num_folds: int = None,
        save_cv_images: bool = True,
        save_test_images: bool = True,
):
    """
    Convert dataset to nnUNet format, see:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md
    """

    if output_folds:
        assert num_folds is not None, "Must specify number of cross validation folds."

    train_image_path = os.path.join(output_path, 'imagesTr')
    train_label_path = os.path.join(output_path, 'labelsTr')
    test_image_path = os.path.join(output_path, 'imagesTs')
    for folder in (train_image_path, train_label_path, test_image_path):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save_images(image_path, subject_id, subject, subject_name_cache, save_images=True, save_label_map=False):
        assert all(image_name in subject for image_name in image_names)

        new_subject_name = f'{short_name}_{subject_id:03}'
        subject_name_cache[subject['name']] = new_subject_name

        if not save_images:
            return

        channel_id = 0
        for image_name in image_names:
            image = subject[image_name]

            for image_channel in image.data.split(1):
                out_image = copy.deepcopy(image)
                out_image.set_data(image_channel)

                out_file_name = f'{new_subject_name}_{channel_id:04}.nii.gz'
                out_path = os.path.join(image_path, out_file_name)
                out_image.save(out_path)
                print("saved", out_path)

                channel_id += 1

        if save_label_map:
            assert label_map_name in subject

            label_map = subject[label_map_name]
            out_file_name = f"{new_subject_name}.nii.gz"
            out_path = os.path.join(train_label_path, out_file_name)
            label_map.save(out_path)
            print("saved", out_path)

    subject_id = 1

    cv_subject_names = {}
    for subject in cross_validation_dataset:
        save_images(train_image_path, subject_id, subject, cv_subject_names, save_images=save_cv_images,
                    save_label_map=True)
        subject_id += 1

    test_subject_names = {}
    if test_dataset is not None:
        for subject in test_dataset:
            save_images(test_image_path, subject_id, subject, test_subject_names, save_images=save_test_images,
                        save_label_map=False)
            subject_id += 1

    label_values = cross_validation_dataset[0][label_map_name]['label_values']
    label_values = {"background": 0, **label_values}

    if metadata is None:
        metadata = {}

    output_path = Path(output_path)
    with open(output_path / "dataset.json", 'w') as f:
        out = {
            'name': short_name,
            **({} if metadata is None else metadata),
            'tensorImageSize': "4D",
            "modality": {str(i): image_name for i, image_name in enumerate(image_names)},
            "labels": {str(label_value): label_name for label_name, label_value in label_values.items()},
            "numTraining": len(cross_validation_dataset),
            "numTest": len(test_dataset) if test_dataset is not None else 0,
            "training": [
                {
                    "image": f'./imagesTr/{new_subject_name}.nii.gz',
                    "label": f'./labelsTr/{new_subject_name}.nii.gz'
                }
                for new_subject_name in cv_subject_names.values()
            ],
            "test": [] if test_dataset is None else [
                f"./imagesTs/{new_subject_name}.nii.gz"
                for new_subject_name in test_subject_names.values()
            ]
        }
        json.dump(out, f, indent=4)

    with open(output_path / "original_subject_names.json", 'w') as f:
        out = {
            "cross_validation_subjects": cv_subject_names,
            "test_subjects": test_subject_names,
        }
        json.dump(out, f, indent=4)

    if output_folds:
        out = [
            {
                'train': [
                    cv_subject_names[subject['name']]
                    for subject in cross_validation_dataset.subjects
                    if subject['fold'] != fold
                ],
                'val': [
                    cv_subject_names[subject['name']]
                    for subject in cross_validation_dataset.subjects
                    if subject['fold'] == fold
                ]
            }
            for fold in range(num_folds)
        ]

        with open(output_path / "cross_validation_splits.json", "w") as f:
            json.dump(out, f, indent=4)

        # nnUNet wants the splits as OrderedDicts, and the list of strings as numpy arrays
        # put this file in the root folder for this task inside nnUNet_preprocessed
        out = [
            OrderedDict({k: np.array(v) for k, v in normal_dict.items()})
            for normal_dict in out
        ]
        with open(output_path / "splits_final.pkl", 'wb') as f:
            pickle.dump(out, f)
