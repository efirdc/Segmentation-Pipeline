import copy

import torch
import fire

from torch.utils.data import Dataset
import torchio as tio

from segmentation_pipeline import *
from research.dmri_hippo.configs.main_config import get_context


class SaggitalSplitWrapper(Dataset):
    def __init__(self, dataset: SubjectFolder):
        self.dataset = dataset

        self.subjects = []
        for subject in dataset.subjects:
            left_subject = copy.deepcopy(subject)
            right_subject = copy.deepcopy(subject)
            left_subject['name'] = f"{subject['name']}_left"
            right_subject['name'] = f"{subject['name']}_right"
            self.subjects += [left_subject, right_subject]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        subject = copy.deepcopy(subject)
        subject.load()
        subject = self.dataset.transform(subject)

        if subject['name'].endswith("left"):
            subject = tio.Crop(cropping=(48, 0, 0, 0, 0, 0))(subject)
        elif subject['name'].endswith('right'):
            subject = tio.Crop(cropping=(0, 48, 0, 0, 0, 0))(subject)
            subject = tio.Flip(axes=(0,))(subject)
        else:
            raise RuntimeError()

        return subject


def main(
        dataset_path: PathLike,
        output_path: PathLike,
        split_and_mirror: bool = False
):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    variables = dict(DATASET_PATH=dataset_path)
    context = get_context(device, variables)
    context.init_components()

    dataset: SubjectFolder = context.dataset
    cross_validation_filter = dataset.cohorts['cross_validation']
    test_filter = NegateFilter(cross_validation_filter)

    cross_validation_dataset = dataset.get_cohort_dataset(cross_validation_filter)
    test_dataset = dataset.get_cohort_dataset(test_filter)

    if split_and_mirror:
        transform = tio.Compose([
            EnforceConsistentAffine(),
            tio.CropOrPad((96, 88, 20), padding_mode='minimum', mask_name='whole_roi_union'),
            CustomRemapLabels(remapping=[("right_whole", 2, 1)], masking_method="Right", include="whole_roi"),
        ])
        cross_validation_dataset.set_transform(transform)
        test_dataset.set_transform(transform)
        cross_validation_dataset = SaggitalSplitWrapper(cross_validation_dataset)
        test_dataset = SaggitalSplitWrapper(test_dataset)
    else:
        transform = EnforceConsistentAffine()
        cross_validation_dataset.set_transform(transform)
        test_dataset.set_transform(transform)

    save_dataset_as_nn_unet(
        cross_validation_dataset=cross_validation_dataset,
        output_path=output_path,
        short_name="DMRI",
        image_names=['mean_dwi', 'md', 'fa'],
        label_map_name='whole_roi',
        test_dataset=test_dataset,
        output_folds=True,
        num_folds=5,
        image_names_to_save=None
    )


if __name__ == "__main__":
    fire.Fire(main)
