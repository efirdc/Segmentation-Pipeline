from typing import Tuple
from random import shuffle

import torch
from torchio.transforms.augmentation import RandomTransform
from torchio.transforms.spatial_transform import SpatialTransform


class PermuteDimensions(SpatialTransform):
    def __init__(
            self,
            permutation: Tuple[int, int, int],
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.permutation = permutation
        self.args_names = ('permutation',)
        self.kwargs = kwargs

    def apply_transform(self, subject):

        # Add channel dimension to the permuation
        permutation = (0,) + tuple([p + 1 for p in self.permutation])

        for image in self.get_images(subject):
            image_data = image['data']
            image_data = image_data.permute(permutation)
            image.set_data(image_data)

        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        permutation = torch.tensor(self.permutation)

        # argsort gets the inverse permutation
        inverse_permutation = torch.argsort(permutation)
        inverse_permutation = tuple(inverse_permutation.tolist())

        return PermuteDimensions(permutation=inverse_permutation, **self.kwargs)


class RandomPermuteDimensions(RandomTransform, SpatialTransform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        permutation = [0, 1, 2]
        shuffle(permutation)
        permutation = tuple(permutation)
        random_permute_dimensions = PermuteDimensions(permutation)
        subject = random_permute_dimensions(subject)
        return subject
