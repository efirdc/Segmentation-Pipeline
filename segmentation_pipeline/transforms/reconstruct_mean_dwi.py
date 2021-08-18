import copy
from typing import Tuple, Union
from numbers import Number

import torchio as tio
from torchio.transforms.augmentation import RandomTransform
import torch
import numpy as np


class ReconstructMeanDWI(RandomTransform):
    def __init__(
            self,
            full_dwi_image_name: str = "full_dwi",
            mean_dwi_image_name: str = "mean_dwi",
            bvec_name: str = "grad",
            num_dwis: Union[int, Tuple[int, int]] = 15,
            num_directions: Union[int, Tuple[int, int]] = 1,
            directionality: Union[Number, Tuple[Number, Number]] = 4,
            bval_range: Tuple[Number, Number] = (1e-5, 501.0),
            **kwargs
    ):
        super().__init__(**kwargs)

        self.full_dwi_image_name = full_dwi_image_name
        self.mean_dwi_image_name = mean_dwi_image_name
        self.bvec_name = bvec_name
        self.num_dwis = num_dwis
        self.num_directions = num_directions
        self.directionality = directionality
        self.bval_range = bval_range
        self.args_names = ("full_dwi_image_name", "mean_dwi_image_name", "bvec_name", "num_dwis", "num_directions",
                           "directionality", "bval_range")

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.full_dwi_image_name not in subject:
            return subject

        full_dwi_image = subject[self.full_dwi_image_name]
        full_dwi = full_dwi_image.data.numpy()
        grad = full_dwi_image[self.bvec_name].numpy()

        bvals = grad[:, 3]
        bvecs = grad[:, :3]
        mask = (bvals > self.bval_range[0]) & (bvals < self.bval_range[1])

        bvecs = bvecs[mask]
        full_dwi = full_dwi[mask]

        num_dwis = self.get_num_dwis()
        num_directions = self.get_num_directions()
        directionality = self.get_directionality()

        random_directions = np.random.randn(3, num_directions)
        random_directions = random_directions / np.linalg.norm(random_directions, axis=0, keepdims=True)

        sample_probabilities = np.max(np.abs(bvecs @ random_directions) ** directionality, axis=1)
        sample_probabilities = sample_probabilities / sample_probabilities.sum()

        indices = np.arange(full_dwi.shape[0])
        indices = np.random.choice(indices, size=num_dwis, p=sample_probabilities)

        mean_dwi = np.mean(full_dwi[indices], axis=0, keepdims=True)

        if self.mean_dwi_image_name in subject:
            mean_dwi_image = subject[self.mean_dwi_image_name]
        else:
            mean_dwi_image = copy.deepcopy(full_dwi_image)
            subject.add_image(mean_dwi_image, self.mean_dwi_image_name)

        mean_dwi_image.set_data(mean_dwi)

        return subject

    def get_num_dwis(self):
        if isinstance(self.num_dwis, int):
            return self.num_dwis
        elif isinstance(self.num_dwis, Tuple):
            low, high = self.num_dwis
            sample = np.random.rand()
            sample = sample ** 3
            sample = sample * (high - low + 1) + low
            sample = int(sample)
            return sample
        else:
            raise ValueError(f"Unexpected type {type(self.num_dwis)} for num_dwis")

    def get_num_directions(self):
        if isinstance(self.num_directions, int):
            return self.num_dwis
        elif isinstance(self.num_directions, Tuple):
            return np.random.randint(self.num_directions[0], self.num_directions[1] + 1)
        else:
            raise ValueError(f"Unexpected type {type(self.num_directions)} for num_directions.")

    def get_directionality(self):
        if isinstance(self.directionality, Number):
            return self.directionality
        elif isinstance(self.directionality, Tuple):
            return np.random.uniform(self.directionality[0], self.directionality[1])
        else:
            raise ValueError(f"Unexpected type {type(self.directionality)} for directionality")

    def is_invertible(self):
        return False


class ReconstructMeanDWIClassic(RandomTransform):
    """Reconstructs Mean Diffusion Weighted Images. `subset_size` gradients are first selected based
    on their distance to a randomly chosen gradient direction. A random number of images in this subset
    are averaged.

    Args:
        bvec_name: Key for the bvec Tensor in the image dictionary
        subset_size: Upper bound of the uniform random variable of images to average

    """

    def __init__(
            self,
            full_dwi_image_name: str = "full_dwi",
            mean_dwi_image_name: str = "mean_dwi",
            bvec_name: str = "grad",
            subset_size: int = 15,
            bval_range: Tuple[float, float] = (1e-5, 501.0),
            **kwargs
    ):
        super().__init__(**kwargs)

        self.full_dwi_image_name = full_dwi_image_name
        self.mean_dwi_image_name = mean_dwi_image_name
        self.bvec_name = bvec_name
        self.subset_size = subset_size
        self.bval_range = bval_range
        self.args_names = ("full_dwi_image_name", "mean_dwi_image_name", "bvec_name", "subset_size", "bval_range")

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if self.full_dwi_image_name not in subject:
            return subject

        full_dwi_image = subject[self.full_dwi_image_name]
        full_dwi = full_dwi_image.data
        grad = full_dwi_image[self.bvec_name]

        bvals = grad[:, 3]
        bvecs = grad[:, :3]
        mask = (bvals > self.bval_range[0]) & (bvals < self.bval_range[1])

        bvecs = bvecs[mask]
        full_dwi = full_dwi[mask]

        rand_bvec = bvecs[np.random.randint(bvecs.shape[0])]
        dist = torch.sum((bvecs - rand_bvec) ** 2, dim=1)
        closest_indices = np.argsort(dist)[: self.subset_size]

        number_of_selections = np.random.randint(low=1, high=self.subset_size)
        ids = torch.randperm(closest_indices.shape[0])[:number_of_selections]
        selected_indices = closest_indices[ids]
        mean_dwi = torch.mean(full_dwi[selected_indices], dim=0)

        if self.mean_dwi_image_name in subject:
            mean_dwi_image = subject[self.mean_dwi_image_name]
        else:
            mean_dwi_image = copy.deepcopy(full_dwi_image)
            subject.add_image(mean_dwi_image, self.mean_dwi_image_name)

        mean_dwi_image.set_data(mean_dwi.unsqueeze(0))

        return subject

    def is_invertible(self):
        return False
