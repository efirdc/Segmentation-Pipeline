import importlib.util
from pathlib import Path
import torch
import time
from typing import Type, Sequence

import torchio as tio
import pandas as pd
from PIL import Image
import wandb


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
        self.start()

    def start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.timestamps = {}

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



def to_wandb(elem):
    if isinstance(elem, dict):
        return {
            key: to_wandb(val)
            for key, val in elem.items()
        }
    elif isinstance(elem, pd.DataFrame):
        return wandb.Table(dataframe=elem)
    elif isinstance(elem, Image.Image):
        return wandb.Image(elem)
    return elem
