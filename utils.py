import os
import time
import copy
import json
import importlib.util
from pathlib import Path
from typing import Type, Sequence, Any, Dict
from abc import ABC, abstractmethod

import torch
import torchio as tio

from transforms import CustomSequentialLabels


def is_sequence(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def as_list(x):
    if x is None:
        return []
    if is_sequence(x):
        return list(x)
    return [x]


def as_set(x):
    if x is None:
        return {}
    if is_sequence(x):
        return set(x)
    return {x}


# Apply this to variable *args so that they can also be optionally given as a sequence
# i.e. some_func(1, 2, 3) can be the same as some_func([1, 2, 3]) if fix_vargs is applied
def vargs_or_sequence(args):
    if is_sequence(args) and len(args) == 1 and is_sequence(args[0]):
        return args[0]
    return args


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def slice_volume(x: torch.tensor, channel_id: int, plane: str, slice_id: int):
    if plane not in ("Axial", "Coronal", "Saggital"):
        raise ValueError(f'plane must be one of "Axial", "Coronal", or "Saggital" not {plane}')
    if plane == "Axial":
        return x[channel_id, :, :, slice_id]
    elif plane == "Coronal":
        return torch.rot90(x[channel_id, :, slice_id, :])
    elif plane == "Saggital":
        return torch.rot90(x[channel_id, slice_id, :, :])


class Timer:
    def __init__(self, device):
        self.device = device

        self.start_time = 0
        self.last_time = 0
        self.timestamps = {}

    def start(self):
        self.start_time = self.last_time = time.time()
        self.timestamps = {stamp: 0.0 for stamp in self.timestamps.keys()}

    def stamp(self, name=None, from_start=False):
        if self.device.type != "cpu":
            torch.cuda.current_stream().synchronize()
    
        new_time = time.time()
        if not from_start:
            dt = new_time - self.last_time
        else:
            dt = time.time() - self.start_time
        self.last_time = new_time
        if name:
            self.timestamps[name] = dt
        return dt


def dict_to_device(elem, device):
    if isinstance(elem, dict):
        return {
            key: dict_to_device(value, device)
            for key, value in elem.items()
        }
    elif isinstance(elem, torch.Tensor):
        tensor = elem
        tensor = tensor.to(device)
        return tensor
    return elem


# TODO: Contribute this as a method to tio.Compose.
# as it doesnt make sense to call unless tio.Compose is the root transform
def filter_transform(
        transform: tio.Compose,
        include_types: Sequence[Type[tio.Transform]] = None,
        exclude_types: Sequence[Type[tio.Transform]] = None,
        ):
    if isinstance(transform, tio.Compose):
        return tio.Compose([
            filter_transform(t, include_types=include_types, exclude_types=exclude_types)
            for t in transform
            if isinstance(t, tio.Compose) or (
                (include_types is None or any(isinstance(t, typ) for typ in include_types))
                and
                (exclude_types is None or not any(isinstance(t, typ) for typ in exclude_types))
            )
        ])
    return transform


def save_dataset_as_nn_unet(dataset, output_path, short_name, image_names, label_map_name,
                            train_cohort, test_cohort=None, metadata=None, fix_affine=False):
    train_image_path = os.path.join(output_path, 'imagesTr')
    train_label_path = os.path.join(output_path, 'labelsTr')
    test_image_path = os.path.join(output_path, 'imagesTs')
    for folder in (train_image_path, train_label_path, test_image_path):
        if not os.path.exists(folder):
            os.makedirs(folder)

    train_dataset = dataset.get_cohort_dataset(train_cohort)

    def save_images(image_path, subject_id, subject):
        channel_id = 0
        for image_name in image_names:
            image = subject[image_name]
            if fix_affine:
                image.affine = image.affine

            for image_channel in image.data.split(1):
                out_image = copy.deepcopy(image)
                out_image.set_data(image_channel)

                out_file_name = f'{short_name}_{subject_id:03}_{channel_id:04}.nii.gz'
                out_image.save(os.path.join(image_path, out_file_name))

                channel_id += 1

    subject_id = 0
    for subject in train_dataset.all_subjects:
        subject_id += 1

        assert all(image_name in subject for image_name in image_names)
        assert label_map_name in subject

        save_images(train_image_path, subject_id, subject)

        label_map = subject[label_map_name]
        label_map = CustomSequentialLabels()(label_map)

        if fix_affine:
            label_map.affine = subject[image_names[0]].affine

        out_file_name = f"{short_name}_{subject_id:03}.nii.gz"
        label_map.save(os.path.join(train_label_path, out_file_name))

    if test_cohort is not None:
        test_dataset = dataset.get_cohort_dataset(test_cohort)

        for subject in test_dataset.all_subjects:
            subject_id += 1

            assert all(image_name in subject for image_name in image_names)

            save_images(test_image_path, subject_id, subject)

    label_values = train_dataset.all_subjects[0][label_map_name]['label_values']
    label_values = {"background": 0, **label_values}

    if metadata is None:
        metadata = {}

    json_dict = {
        'name': short_name,
        **({} if metadata is None else metadata),
        'tensorImageSize': "4D",
        "modality": {str(i): image_name for i, image_name in enumerate(image_names)},
        "labels": {str(label_value): label_name for label_name, label_value in label_values.items()},
        "numTraining": len(train_dataset),
        "numTest": len(test_dataset) if test_cohort is not None else 0,
        "training": [
            {
                "image": f'./imagesTr/{short_name}_{i:03}.nii.gz',
                "label": f'./labelsTr/{short_name}_{i:03}.nii.gz'
            }
            for i in range(1, len(train_dataset) + 1)
        ],
        "test": [] if test_cohort is None else [
            f"./imagesTs/{short_name}_{i:03}.nii.gz"
            for i in range(len(train_dataset) + 1, len(train_dataset) + len(test_dataset) + 1)
        ]
    }

    json_path = os.path.join(output_path, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=4)


def dont_collate(subjects):
    return subjects


def collate_subjects(subjects: Sequence[tio.Subject], image_names: Sequence[str], device: torch.device):
    batch = {}
    for image_name in image_names:
        data = torch.stack([subject[image_name]["data"] for subject in subjects])
        data = data.to(device)
        batch[image_name] = data
    return batch


class Config(ABC):
    """Representation of a class that has configuration to be stored"""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        raise NotImplementedError()
