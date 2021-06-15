import importlib.util
from pathlib import Path
import torch
import time
from collections.abc import Sequence


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


class CudaTimer:
    def __init__(self):
        self.start()

    def start(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.timestamps = {}

    def stamp(self, name=None, from_start=False):
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
