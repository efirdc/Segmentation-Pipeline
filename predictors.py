from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torchio as tio
from torch import nn
from torch.utils.data import DataLoader
from torchio.typing import TypePatchSize

from transforms import *
from utils import Config, collate_subjects, dont_collate


class SegPredictor(ABC, Config):
    """Representation to predict segmentations"""

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        subjects: Sequence[tio.Subject],
        label_attributes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Sequence[tio.Subject], Dict[str, torch.Tensor]]:
        """Creates predictions for subjects and adds the predictions as an image with name 'y_pred' and
        batch with with key 'y_pred'"""


class StandardPredict(SegPredictor):
    """ Creates predictions on whole images"""

    def __init__(self, device: torch.device, sagittal_split: bool = False, image_names=("X",)):
        self.device = device
        self.sagittal_split = sagittal_split
        self.image_names = image_names

    def predict(self, model, subjects, label_attributes=None):

        batch = collate_subjects(subjects, image_names=self.image_names, device=self.device)

        if label_attributes is None:
            label_attributes = {}

        if self.sagittal_split:
            # apply split and flip on batch to reduce memory
            split = self.split_and_flip(batch['X'])
            out = model(split)
            out = self.reverse_split_and_flip(out)
        else:
            out = model(batch["X"])

        batch['y_pred'] = out

        out_subjects = []
        for i in range(len(subjects)):
            subject = subjects[i]
            y_pred = tio.LabelMap(tensor=batch["y_pred"][i].detach().cpu(), **label_attributes)
            subject.add_image(y_pred, "y_pred")
            subject = EnforceConsistentAffine(source_image_name="X")(subject)
            out_subjects.append(subject)

        return out_subjects, batch

    def split_and_flip(self, x: torch.Tensor) -> torch.Tensor:
        x_split = list(x.split(x.shape[2] // 2, dim=2))
        x_split[1] = x_split[1].flip(2)
        x = torch.cat(x_split, dim=0)
        return x

    def reverse_split_and_flip(self, x: torch.Tensor) -> torch.Tensor:
        x_split = list(x.split(x.shape[0] // 2, dim=0))
        x_split[1] = x_split[1].flip(2)
        x = torch.cat(x_split, dim=2)
        return x


class PatchPredict(SegPredictor):
    """ Creates predictions on patches and aggregates"""

    def __init__(
        self,
        device: torch.device,
        patch_batch_size: int = 16,
        patch_size: TypePatchSize = None,
        patch_overlap: TypePatchSize = (0, 0, 0),
        padding_mode: Union[str, float, None] = None,
        overlap_mode: str = "average",
        image_names: Sequence[str] = ("X",),
    ):
        self.device = device
        self.patch_batch_size = patch_batch_size
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.padding_mode = padding_mode
        self.overlap_mode = overlap_mode
        self.image_names = image_names

    def predict(self, model, subjects, label_attributes=None):

        if label_attributes is None:
            label_attributes = {}

        out_subjects = []
        batch = {}
        for subject in subjects:
            grid_sampler = tio.GridSampler(subject, self.patch_size, self.patch_overlap, self.padding_mode)
            patch_loader = DataLoader(grid_sampler, batch_size=self.patch_batch_size, collate_fn=dont_collate)
            aggregator = tio.GridAggregator(grid_sampler, overlap_mode=self.overlap_mode)

            for subject_patches in patch_loader:
                locations = torch.stack([patch["location"] for patch in subject_patches])
                patch_batch = collate_subjects(subject_patches, self.image_names, device=self.device)
                with torch.no_grad():
                    y_pred_patch = model(patch_batch["X"])
                aggregator.add_batch(y_pred_patch, locations)

            aggregated_patch = aggregator.get_output_tensor().cpu()
            y_pred = tio.LabelMap(tensor=aggregated_patch, **label_attributes)
            subject.add_image(y_pred, "y_pred")
            subject = EnforceConsistentAffine(source_image_name="X")(subject)
            out_subjects.append(subject)

        batch = collate_subjects(subjects, image_names=self.image_names, device=self.device)
        batch["y_pred"] = torch.stack([subject["y_pred"]["data"] for subject in out_subjects])

        return out_subjects, batch
