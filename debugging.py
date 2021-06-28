import torch
import torchio as tio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

from transforms import *
from utils import load_module, filter_transform

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    #config = load_module("./configs/diffusion_hippocampus.py")
    #variables = dict(DATASET_PATH="X:/Datasets/Diffusion_MRI/Subjects/", CHECKPOINTS_PATH="X:/Checkpoints/")
    #context = config.get_context(device, variables)

    config = load_module("./configs/diffusion_hippocampus.py")
    variables = dict(DATASET_PATH="X:/Datasets/Diffusion_MRI/", CHECKPOINTS_PATH="X:/Checkpoints/")
    context = config.get_context(device, variables)

    untransformed_subject = context.dataset.subjects[0]
    print("Original labels:")
    print(untransformed_subject['whole_roi']['label_values'])

    subject = context.dataset[0]
    print("\nTransformed labels:")
    print(subject['y']['label_values'])

    transform = subject.get_composed_history()
    label_transform_types = [LabelTransform, CopyProperty, RenameProperty, ConcatenateImages]
    label_transform = filter_transform(transform, include_types=label_transform_types)
    inverse_label_transform = label_transform.inverse(warn=False)

    inverse_subject = inverse_label_transform(tio.Subject({'y': subject['y']}))
    print("\nInverse labels")
    print(inverse_subject['whole_roi']['label_values'])

    evaluation_transform = tio.Compose([
        CustomSequentialLabels(),
        filter_transform(inverse_label_transform, exclude_types=[CustomRemapLabels]).inverse(warn=False)
    ])
    print(evaluation_transform)

    evaluation_subject = evaluation_transform(inverse_subject)
    print("\nEvaluation labels")
    print(evaluation_subject['y']['label_values'])



