import copy
import os
import sys
from pathlib import Path

import torch
import torchio as tio

from segmentation_pipeline import *


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    msseg_ensemble_02_path = Path("X:\\Checkpoints\\MSSEG2\\cross_validation_01\\ensemble_02\\")
    msseg_fold_paths = list(msseg_ensemble_02_path.iterdir())

    variables = dict(DATASET_PATH="X:/Datasets/MSSEG2_resampled/")
    device = torch.device('cuda')
    context = TorchContext(device, file_path=msseg_fold_paths[0], variables=variables)
    context.init_components()
