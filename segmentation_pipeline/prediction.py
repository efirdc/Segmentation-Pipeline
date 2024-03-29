import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torchio as tio
from torch import nn
from torch.utils.data import DataLoader
from torchio.typing import TypeSpatialShape
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

from .transforms import *
from .utils import Config, collate_subjects, no_op


def split_and_flip(x: torch.Tensor) -> torch.Tensor:
    x_split = list(x.split(x.shape[2] // 2, dim=2))
    x_split[1] = x_split[1].flip(2)
    x = torch.cat(x_split, dim=0)
    return x


def reverse_split_and_flip(x: torch.Tensor) -> torch.Tensor:
    x_split = list(x.split(x.shape[0] // 2, dim=0))
    x_split[1] = x_split[1].flip(2)
    x = torch.cat(x_split, dim=2)
    return x


def apply_stochastic_matrix(y_pred, y_prior):
    N = y_prior.shape[0]
    C = y_prior.shape[1]
    spatial_shape = y_prior.shape[2:]

    y_pred = y_pred.reshape(N, C, C, *spatial_shape)
    y_pred = (y_pred * y_prior[:, None]).sum(dim=1)

    return y_pred


class Predictor(ABC, Config):
    """Representation to get model predictions"""

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        device: torch.device,
        subjects: Sequence[tio.Subject],
        label_attributes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Sequence[tio.Subject], Dict[str, torch.Tensor]]:
        """Creates predictions for subjects and adds the predictions as an image with name 'y_pred' and
        batch with with key 'y_pred'"""
        raise NotImplementedError()


class StandardPredict(Predictor):
    """ Creates predictions on whole images"""

    def __init__(
            self,
            image_names: Sequence[str] = ("X",),
            sagittal_split: bool = False,
            refine_image: str = None,
    ):
        image_names = list(image_names)
        if refine_image is not None and refine_image not in image_names:
            image_names.append(refine_image)
        self.image_names = image_names
        self.sagittal_split = sagittal_split
        self.refine_image = refine_image

    def predict(self, model, device, subjects, label_attributes=None):

        batch = collate_subjects(subjects, image_names=self.image_names, device=device)

        if label_attributes is None:
            label_attributes = {}

        if self.sagittal_split:
            split = split_and_flip(batch['X'])
            y_pred = model(split)
            y_pred = reverse_split_and_flip(y_pred)
        else:
            y_pred = model(batch["X"])

        #if self.refine_image is not None:
        #    y_prior = batch[self.refine_image]
        #    y_pred = apply_stochastic_matrix(y_pred, y_prior)
        #    y_pred = (y_pred * y_prior[:, None]).sum(dim=1)

        batch['y_pred'] = y_pred

        out_subjects = []
        for i in range(len(subjects)):
            subject = subjects[i]
            y_pred = tio.LabelMap(tensor=batch["y_pred"][i].detach().cpu(), **copy.deepcopy(label_attributes))
            subject.add_image(y_pred, "y_pred")
            subject = EnforceConsistentAffine(source_image_name="X")(subject)
            out_subjects.append(subject)

        return out_subjects, batch


class PatchPredict(Predictor):
    """ Creates predictions on patches and aggregates"""

    def __init__(
        self,
        image_names: Sequence[str] = ("X",),
        patch_batch_size: int = 16,
        patch_size: TypeSpatialShape = None,
        patch_overlap: TypeSpatialShape = (0, 0, 0),
        padding_mode: Union[str, float, None] = None,
        overlap_mode: str = "average",
    ):
        self.image_names = image_names
        self.patch_batch_size = patch_batch_size
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.padding_mode = padding_mode
        self.overlap_mode = overlap_mode

    def predict(self, model, device, subjects, label_attributes=None):

        if label_attributes is None:
            label_attributes = {}

        out_subjects = []
        batch = {}
        for subject in subjects:
            grid_sampler = tio.GridSampler(subject, self.patch_size, self.patch_overlap, self.padding_mode)
            patch_loader = DataLoader(grid_sampler, batch_size=self.patch_batch_size, collate_fn=no_op)
            aggregator = tio.GridAggregator(grid_sampler, overlap_mode=self.overlap_mode)

            for subject_patches in patch_loader:
                locations = torch.stack([patch["location"] for patch in subject_patches])
                patch_batch = collate_subjects(subject_patches, self.image_names, device=device)
                with torch.no_grad():
                    y_pred_patch = model(patch_batch["X"])
                aggregator.add_batch(y_pred_patch, locations)

            aggregated_patch = aggregator.get_output_tensor().cpu()
            y_pred = tio.LabelMap(tensor=aggregated_patch, **copy.deepcopy(label_attributes))
            subject.add_image(y_pred, "y_pred")
            subject = EnforceConsistentAffine(source_image_name="X")(subject)
            out_subjects.append(subject)

        batch = collate_subjects(subjects, image_names=self.image_names, device=device)
        batch["y_pred"] = torch.stack([subject["y_pred"]["data"] for subject in out_subjects])

        return out_subjects, batch


def add_evaluation_labels(subjects: Sequence[tio.Subject]):
    for subject in subjects:
        transform = subject.get_composed_history()
        label_transform_types = [LabelTransform, CopyProperty, RenameProperty, ConcatenateImages]
        label_transform = filter_transform(transform, include_types=label_transform_types)
        evaluation_transform = label_transform.inverse(warn=False)

        if 'y_pred' in subject:
            pred_subject = tio.Subject({'y': subject['y_pred']})
            y_pred_eval = evaluation_transform(pred_subject).get_first_image()
            subject.add_image(y_pred_eval, 'y_pred_eval')

        if 'y' in subject:
            target_subject = tio.Subject({'y': subject['y']})
            y_eval = evaluation_transform(target_subject).get_first_image()
            subject.add_image(y_eval, 'y_eval')
