import argparse
import os
import json
import copy
from typing import Sequence
from pathlib import Path
from collections import OrderedDict
import pickle

import torch
import numpy as np

from ..transforms import CustomSequentialLabels, EnforceConsistentAffine
from ..utils import load_module


def save_dataset_as_nn_unet(
        dataset,
        output_path: str,
        short_name: str,
        image_names: Sequence[str],
        label_map_name: str,
        cross_validation_cohort: str,
        test_cohort: str = None,
        metadata: dict = None,
        output_folds: bool = False,
        num_folds: int = None,
        skip_saving_images: bool = False,
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

    cross_validation_dataset = dataset.get_cohort_dataset(cross_validation_cohort)


    def save_images(image_path, subject_id, subject, subject_name_cache, save_label_map=False):
        assert all(image_name in subject for image_name in image_names)

        new_subject_name = f'{short_name}_{subject_id:03}'
        subject_name_cache[subject['name']] = new_subject_name

        if skip_saving_images:
            return

        subject = copy.deepcopy(subject)
        subject.load()
        subject = EnforceConsistentAffine(source_image_name=image_names[0])(subject)

        channel_id = 0
        for image_name in image_names:
            image = subject[image_name]

            for image_channel in image.data.split(1):
                out_image = copy.deepcopy(image)
                out_image.set_data(image_channel)

                out_file_name = f'{new_subject_name}_{channel_id:04}.nii.gz'
                out_image.save(os.path.join(image_path, out_file_name))

                channel_id += 1

        if save_label_map:
            assert label_map_name in subject

            label_map = subject[label_map_name]
            label_map = CustomSequentialLabels()(label_map)
            out_file_name = f"{new_subject_name}.nii.gz"
            label_map.save(os.path.join(train_label_path, out_file_name))

    subject_id = 1

    cv_subject_names = {}
    for subject in cross_validation_dataset.all_subjects:
        save_images(train_image_path, subject_id, subject, cv_subject_names, save_label_map=True)
        subject_id += 1

    test_subject_names = {}
    if test_cohort is not None:
        test_dataset = dataset.get_cohort_dataset(test_cohort)
        for subject in test_dataset.all_subjects:
            save_images(test_image_path, subject_id, subject, test_subject_names, save_label_map=False)
            subject_id += 1

    label_values = cross_validation_dataset.all_subjects[0][label_map_name]['label_values']
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
            "numTest": len(test_dataset) if test_cohort is not None else 0,
            "training": [
                {
                    "image": f'./imagesTr/{new_subject_name}.nii.gz',
                    "label": f'./labelsTr/{new_subject_name}.nii.gz'
                }
                for new_subject_name in cv_subject_names.values()
            ],
            "test": [] if test_cohort is None else [
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a dataset to nnUnet format")
    parser.add_argument("config", type=str)
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("short_name", type=str)
    parser.add_argument("image_names", type=str, help='List of image names, separated by commas. ex: "mean_dwi,md,fa"')
    parser.add_argument("label_map_name", type=str)
    parser.add_argument("cross_validation_cohort", type=str)
    parser.add_argument("--test_cohort", type=str, default=None)
    parser.add_argument("--output_folds", default=False, action="store_true")
    parser.add_argument("--num_folds", type=int, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    config = load_module(args.config)
    variables = dict(DATASET_PATH=args.dataset_path)
    context = config.get_context(device, variables)
    context.init_components()

    save_dataset_as_nn_unet(
        context.dataset,
        args.output_path,
        args.short_name,
        args.image_names.split(','),
        args.label_map_name,
        args.cross_validation_cohort,
        args.test_cohort,
        output_folds=args.output_folds,
        num_folds=args.num_folds,
    )
