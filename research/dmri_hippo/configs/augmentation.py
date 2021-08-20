import os

import torchio as tio

from . import main_config as base_config
from segmentation_pipeline import *


def get_context(
        device,
        variables,
        augmentation_mode,
        **kwargs
):

    context = base_config.get_context(device, variables, **kwargs)
    context.file_paths.append(os.path.abspath(__file__))

    # training_transform is a tio.Compose where the second transform is the augmentation
    dataset_defn = context.get_component_definition("dataset")
    training_transform = dataset_defn['params']['transforms']['training']

    dwi_augmentation = ReconstructMeanDWI(num_dwis=(1, 25), num_directions=(1, 3), directionality=(4, 10))

    noise = tio.RandomNoise(std=0.035, p=0.3)
    blur = tio.RandomBlur((0, 1), p=0.2)
    standard_augmentations = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2)),
        tio.RandomElasticDeformation(p=0.5, num_control_points=(7, 7, 4), locked_borders=1,
                                     image_interpolation='bspline', exclude="full_dwi"),
        tio.RandomBiasField(p=0.5),
        tio.RescaleIntensity((0, 1), (0.01, 99.9)),
        tio.RandomGamma(p=0.8),
        tio.RescaleIntensity((-1, 1)),
        tio.OneOf([
            tio.Compose([blur, noise]),
            tio.Compose([noise, blur]),
        ])
    ], exclude="full_dwi")

    if augmentation_mode == 'no_augmentation':
        training_transform.transforms.pop(1)
    elif augmentation_mode == 'standard':
        training_transform.transforms[1] = standard_augmentations
    elif augmentation_mode == 'dwi_reconstruction':
        training_transform.transforms[1] = dwi_augmentation
    elif augmentation_mode == 'combined':
        training_transform.transforms[1] = tio.Compose([dwi_augmentation, standard_augmentations])
    else:
        raise ValueError(f"Invalid augmentation mode {augmentation_mode}")

    return context
