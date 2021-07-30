from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torchio as tio
from torch import nn
from torch.utils.data import DataLoader
from torchio.typing import TypePatchSize

from transforms import *
from utils import Config, collate_subjects, dont_collate


class SegPredictor(ABC):
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


class StandardPredict(SegPredictor, Config):
    """ Creates predictions on whole images"""

    def __init__(self, device: torch.device, image_names: Sequence[str]):
        self.device = device
        self.image_names = image_names

    def predict(self, model, subjects, label_attributes):

        batch = collate_subjects(subjects, image_names=self.image_names, device=self.device)

        if label_attributes is None:
            label_attributes = {}

        batch["y_pred"] = model(batch["X"])

        out_subjects = []
        for i in range(len(subjects)):
            subject = subjects[i]
            y_pred = tio.LabelMap(tensor=batch["y_pred"][i].detach().cpu(), **label_attributes)
            subject.add_image(y_pred, "y_pred")
            subject = EnforceConsistentAffine(source_image_name="X")(subject)
            out_subjects.append(subject)

        return out_subjects, batch

    def getConfig(self) -> Dict[str, Any]:
        return {}


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
        image_names: Sequence[str] = ["X"],
    ):
        self.device = device
        self.patch_batch_size = patch_batch_size
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.padding_mode = padding_mode
        self.overlap_mode = overlap_mode
        self.image_names = image_names

    def predict(self, model, subjects, label_attributes):

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

    def getConfig(self) -> Dict[str, Any]:
        return {'patch_batch_size': self.patch_overlap, 'padding_mode': self.padding_mode, 'overlap_mode': self.overlap_mode}