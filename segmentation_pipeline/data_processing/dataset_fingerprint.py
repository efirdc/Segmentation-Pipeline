import copy
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torchio as tio

from .subject_folder import SubjectFolder
from ..utils import CompactJSONEncoder


def get_bounds(mask: torch.BoolTensor):
    where = np.where(mask)
    extents = []
    crop = []
    size = []
    center = []

    for i, w in enumerate(where):
        w_min = int(w.min())
        w_max = int(w.max())
        extents += [w_min, w_max]
        crop += [w_min, mask.shape[i] - w_max]
        size.append(w_max - w_min)
        center.append((w_max + w_min) / 2)

    return {"extents": extents, "crop": crop, "size": size, "center": center}


def get_label_bounds(label_map: tio.LabelMap):
    label_bounds = {}
    label_values = label_map['label_values']

    all_mask = (label_map.data != 0)
    label_bounds['all'] = get_bounds(all_mask[0])

    for label_name, label_value in label_values.items():
        mask = (label_map.data == label_value)
        label_bounds[label_name] = get_bounds(mask[0])

    return label_bounds


def get_summary_stats(tensor: torch.Tensor, **kwargs):

    def item(x):
        if not isinstance(x, torch.Tensor):
            x = x.values
        if x.numel() > 1:
            return tuple(x.tolist())
        else:
            return x.item()

    return {
        'mean': item(torch.mean(tensor.float(), **kwargs)),
        'std': item(torch.std(tensor.float(), **kwargs)),
        'median': item(torch.median(tensor, **kwargs)),
        'min': item(torch.min(tensor, **kwargs)),
        'max': item(torch.max(tensor, **kwargs)),
    }


def merge_dict(in_dict: dict, out_dict: dict):
    for k, v in in_dict.items():
        if k not in out_dict:
            if isinstance(v, dict):
                out_dict[k] = {}
                merge_dict(in_dict[k], out_dict[k])
            else:
                out_dict[k] = [v]
        else:
            if isinstance(v, dict):
                merge_dict(in_dict[k], out_dict[k])
            else:
                out_dict[k].append(v)


def summarize(elem):
    if isinstance(elem, dict):
        return {k: summarize(v) for k, v in elem.items()}
    elif isinstance(elem, list):
        return get_summary_stats(torch.tensor(elem), dim=0)
    else:
        raise RuntimeError(f"Unexpected element {elem}")


def get_dataset_fingerprint(
        dataset: SubjectFolder,
        transform: tio.Transform = None,
        save: bool = False,
        image_names: Optional[Sequence[str]] = None
):
    subject_fingerprints = {}

    for subject in dataset.all_subjects:

        if transform is not None:
            subject = copy.deepcopy(subject)
            subject.load()
            subject = transform(subject)

        if image_names is None:
            images = {key: val for key, val in subject.items() if isinstance(val, tio.ScalarImage)}
            label_maps = {key: val for key, val in subject.items() if isinstance(val, tio.LabelMap)}
        else:
            images = {image_name: subject[image_name] for image_name in image_names
                      if image_name in subject and isinstance(subject[image_name], tio.ScalarImage)}
            label_maps = {image_name: subject[image_name] for image_name in image_names
                          if image_name in subject and isinstance(subject[image_name], tio.LabelMap)}

        subject_fingerprints[subject['name']] = {
            'spacing': subject.spacing,
            'spatial_shape': subject.spatial_shape,
            'label_bounds': {name: get_label_bounds(label_map)
                             for name, label_map in label_maps.items()},
            'intensity_stats': {name: get_summary_stats(image.data)
                                for name, image in images.items()}
        }
    fingerprints = list(subject_fingerprints.values())

    if save:
        json_encoder = CompactJSONEncoder(indent=2)
        out_path = Path(dataset.root) / "fingerprint"
        out_path.mkdir(parents=True, exist_ok=True)

        with (out_path / "subject_fingerprints.json").open(mode="w") as f:
            out_str = json_encoder.encode(subject_fingerprints)
            f.write(out_str)

    merged_fingerprint = {}
    for fingerprint in fingerprints:
        merge_dict(fingerprint, merged_fingerprint)
    summary_fingerprint = summarize(merged_fingerprint)

    if save:
        with open(out_path / "fingerprint.json", "w") as f:
            out_str = json_encoder.encode(summary_fingerprint)
            f.write(out_str)

    return subject_fingerprints, summary_fingerprint
