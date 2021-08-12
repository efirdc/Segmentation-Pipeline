import importlib.util
from pathlib import Path
from typing import Sequence
from inspect import signature
from random import Random

import torch
import torchio as tio


def no_op(x):
    return x


def is_sequence(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def as_list(x):
    if x is None:
        return []
    if is_sequence(x):
        return list(x)
    return [x]


def as_tuple(x):
    if x is None:
        return ()
    if is_sequence(x):
        return tuple(x)
    return x,


def as_set(x):
    if x is None:
        return {}
    if is_sequence(x):
        return set(x)
    return {x}


def vargs_or_sequence(args):
    """
    Apply this to variable *args so that they can also be optionally given as a sequence
    i.e. some_func(1, 2, 3) can be the same as some_func([1, 2, 3]) if this is applied
    """
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


def collate_subjects(
        subjects: Sequence[tio.Subject],
        image_names: Sequence[str],
        device: torch.device
):
    batch = {}
    for image_name in image_names:
        data = torch.stack([subject[image_name]["data"] for subject in subjects])
        data = data.to(device)
        batch[image_name] = data
    return batch


def flatten_nested_dict(nested_dict, max_depth=10, flatten_sequences=True):
    """
    Flattens a nested dictionary. (i.e. a dict whose values are other dicts)

    The keys of the nested_dict are joined together by a '.'
    For example if nested_dict['foo']['bar']['baz'] are valid keys for the input nested_dict,
    then flat_dict['foo.bar.baz'] is a valid key for the output flat_dict

    """
    flat_dict = nested_dict.copy()
    for _ in range(max_depth):

        flat = True
        for key, value in flat_dict.copy().items():
            if isinstance(value, dict):
                flat = False
                del flat_dict[key]
                flat_dict.update({
                    f"{key}.{nested_key}": nested_value
                    for nested_key, nested_value in value.items()
                })
            elif is_sequence(value) and flatten_sequences:
                flat = False
                del flat_dict[key]
                flat_dict.update({
                    f"{key}.{i}": nested_value
                    for i, nested_value in enumerate(value)
                })

        if flat:
            break

    return flat_dict


def auto_str(obj):
    """ Automatically generates a string representation for an object.
    with the following syntax:
    ClassName(param_1=arg_1, param_2=arg_2, ... , param_n=arg_n)

    For this to work the parameter arguments must be stored as class properties with the same name.
    If a parameter is not saved it will show up as "param_i=?"
    """
    sig = signature(obj.__init__)
    param_names = list(sig.parameters.keys())
    arg_dict = {}
    for param_name in param_names:
        if param_name not in obj.__dict__:
            arg_dict[param_name] = "?"
        else:
            arg_dict[param_name] = str(obj.__dict__[param_name])
    out_str = ", ".join([f"{param_name}={arg}" for param_name, arg in arg_dict.items()])
    out_str = f"{obj.__class__.__name__}({out_str})"
    return out_str


def random_folds(size, num_folds, seed):
    fold_ids = [i % num_folds for i in range(size)]
    Random(seed).shuffle(fold_ids)
    return fold_ids
