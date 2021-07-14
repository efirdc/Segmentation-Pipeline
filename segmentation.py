import copy
from typing import Sequence, Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torchio as tio
from torchio.typing import TypePatchSize
from torchio.transforms.preprocessing.label.label_transform import LabelTransform
from torchio.data.sampler.sampler import PatchSampler
from torch_context import TorchContext

from utils import filter_transform
from transforms import *


def dont_collate(subjects):
    return subjects


def collate_subjects(
        subjects: Sequence[tio.Subject],
        image_names: Sequence[str],
        device: torch.device
):
    batch = {}
    for image_name in image_names:
        data = torch.stack([subject[image_name]['data'] for subject in subjects])
        data = data.to(device)
        batch[image_name] = data
    return batch


def seg_predict(
        model: nn.Module,
        batch: Dict[str, torch.tensor],
        subjects: Sequence[tio.Subject],
        y_sample: tio.LabelMap
):
    batch['y_pred'] = model(batch['X'])

    for i in range(len(subjects)):
        subject = subjects[i]
        y_pred = copy.deepcopy(y_sample)
        y_pred.set_data(batch['y_pred'][i].detach().cpu())
        subject['y_pred'] = y_pred


def patch_predict(
        model: nn.Module,
        subjects: Sequence[tio.Subject],
        y_sample: tio.LabelMap,
        patch_batch_size: int = 16,
        patch_size: TypePatchSize = None,
        patch_overlap: TypePatchSize = (0, 0, 0),
        padding_mode: Union[str, float, None] = None,
        overlap_mode: str = 'average',
):
    for subject in subjects:
        grid_sampler = tio.GridSampler(subject,
                                       patch_size,
                                       patch_overlap,
                                       padding_mode)
        patch_loader = DataLoader(grid_sampler,
                                  batch_size=patch_batch_size,
                                  collate_fn=dont_collate)
        aggregator = tio.GridAggregator(grid_sampler, overlap_mode=overlap_mode)

        for subject_patches in patch_loader:
            locations = torch.stack([patch['location'] for patch in subject_patches])
            batch = collate_subjects(subject_patches, image_names=['X'], device=model.device)
            with torch.no_grad():
                y_pred_patch = model(batch['X'])
            aggregator.add_batch(y_pred_patch, locations)

        aggregated_patch = aggregator.get_output_tensor().cpu()
        y_pred = copy.deepcopy(y_sample)
        y_pred.set_data(aggregated_patch)
        subject['y_pred'] = y_pred


def add_evaluation_labels(subjects: Sequence[tio.Subject]):
    for subject in subjects:
        transform = subject.get_composed_history()
        label_transform_types = [LabelTransform, CopyProperty, RenameProperty, ConcatenateImages]
        label_transform = filter_transform(transform, include_types=label_transform_types)
        inverse_label_transform = label_transform.inverse(warn=False)

        evaluation_transform = tio.Compose([
            inverse_label_transform,
            CustomSequentialLabels(),
            filter_transform(inverse_label_transform, exclude_types=[CustomRemapLabels]).inverse(warn=False)
        ])

        if 'y_pred' in subject:
            pred_subject = tio.Subject({'y': subject['y_pred']})
            subject['y_pred_eval'] = evaluation_transform(pred_subject)['y']

        if 'y' in subject:
            target_subject = tio.Subject({'y': subject['y']})
            subject['y_eval'] = evaluation_transform(target_subject)['y']
