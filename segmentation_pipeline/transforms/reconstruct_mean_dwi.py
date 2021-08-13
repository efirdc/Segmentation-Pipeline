import copy

import torchio as tio
import torch
import numpy as np


class ReconstructMeanDWI(tio.Transform):
    """ Reconstructs Mean Diffusion Weighted Images. `subset_size` gradients are first selected based
    on their distance to a randomly chosen gradient direction. A random number of images in this subset
    are averaged.

    Args:
        bvec_name: Key for the bvec Tensor in the image dictionary
        subset_size: Upper bound of the uniform random variable of images to average

    """    
    def __init__(
            self,
            bvec_name: str,
            subset_size: int,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.bvec_name = bvec_name
        self.subset_size = subset_size
        self.args_names = ('bvec_name', 'subset_size')

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:

        for image in subject.get_images(include=self.include, exclude=self.exclude):
            image_data = image.data
            grad = image[self.bvec_name]

            b500_mask = (grad[:, 3] < 501.) & (grad[:, 3] > 1e-5)

            b500_grads = grad[b500_mask]
            b500s = image_data[b500_mask]

            bvecs = b500_grads[:, :3]

            rand_bvec = bvecs[np.random.randint(bvecs.shape[0])]
            dist = torch.sum((bvecs - rand_bvec) ** 2, dim=1)
            closest_indices = np.argsort(dist)[:self.subset_size]

            number_of_selections = np.random.randint(self.subset_size)
            ids = torch.randperm(closest_indices.shape[0])[:number_of_selections]
            selected_indices = closest_indices[ids]
            averaged = torch.mean(b500s[selected_indices], dim=0)
            image.set_data(averaged.unsqueeze(0))
        return subject

    def is_invertible(self):
        return False


