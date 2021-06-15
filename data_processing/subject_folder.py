import os

import copy
from typing import Sequence

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchio as tio
from torchio.transforms.preprocessing.label.label_transform import LabelTransform

from .subject_loaders import SubjectLoader
from .subject_filters import SubjectFilter


class SubjectFolder(Dataset):
    """ A PyTorch Dataset for 3D medical data.

    The data must be organized so that each subject has their own folder within a root directory.
    Then each subjects folder contains all of the images for the subject.
    An attributes.json file can be optionally included in the subject folder to specify per-subject data
    (i.e. subject age, gender, scanner differences, etc.)

    Args:
        root: Path to the root of the subject folder dataset.
        subject_path: Path to folder containing subjects, relative to the root.
        subject_loader: A SubjectLoader pipeline that loads subject data from the subject folders.
        subject_filter: A SubjectFilter pipeline which selects which subjects will be part of the dataset.
        transforms: A tio.Transform pipeline that is applied to each subject.
        collate_attributes: Subject attributes that will be grouped into tensors by a dataloader.
    """
    def __init__(
            self,
            root: str,
            subject_path: str,
            subject_loader: SubjectLoader,
            subject_filter: SubjectFilter = None,
            transforms: tio.Transform = None,
            collate_attributes: Sequence[str] = None
    ):
        self.root = root
        self.subject_path = os.path.join(self.root, subject_path)
        self.subject_loader = subject_loader
        self.subject_filter = subject_filter
        self.transforms = transforms
        self.collate_attributes = collate_attributes

        self.label_transform = tio.Compose([
            transform for transform in transforms
            if isinstance(transform, LabelTransform)
        ])
        self.inverse_label_transform = self.label_transform.inverse(warn=False)

        # Loops through all subjects in the directory
        self.all_subjects = []
        self.subject_names = os.listdir(self.subject_path)
        for subject_name in self.subject_names:

            # The subject_data dictionary will be used to initialize the tio.Subject
            subject_folder = os.path.join(self.subject_path, subject_name)
            subject_data = dict(name=subject_name, folder=subject_folder)

            # Apply subject loaders
            subject_data = self.subject_loader(subject_data)
            self.all_subjects.append(tio.Subject(**subject_data))

        # Apply the subject_filter
        self.subjects = self.all_subjects
        if self.subject_filter:
            self.subjects = self.subject_filter(self.subjects)
        self.excluded_subjects = list(set(self.all_subjects) - set(self.subjects))

        # Dictionaries to lookup subjects by their names
        self.subjects_map = {subject.name: subject for subject in self.subjects}
        self.all_subjects_map = {subject.name: subject for subject in self.all_subjects}


    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):

        # Get subjects by an integer ID in 0..N, or by the subject's folder name
        if isinstance(idx, int):
            subject = self.subjects[idx]
        elif isinstance(idx, str):
            subject = self.subjects_map[idx]
        else:
            raise ValueError(f"Subject index must be an int or a string, not {idx} of type {type(idx)}")

        # Load subject and apply transform
        subject = copy.deepcopy(subject)
        subject.load()
        if self.transforms is not None:
            subject = self.transforms(subject)

        return subject

    def __contains__(self, item):
        if isinstance(item, int):
            return item < len(self)
        if isinstance(item, str):
            return item in self.subjects_map
        if isinstance(item, tio.Subject):
            return item in self.subjects
        return False

    # Preloads the images for all subjects. Typically they are lazy-loaded in __getitem__.
    def preload_subjects(self):
        for subject in self.subjects:
            subject.load()

    def collate(self, batch: Sequence[tio.Subject]):
        out_dict = default_collate(
            {attrib: subject[attrib] for attrib in self.collate_attributes}
            for subject in batch
        )
        out_dict['subjects'] = batch
        return out_dict

    def load_and_collate_all_subjects(self):
        all_subjects_loaded = [self[i] for i in range(len(self))]
        all_subjects_collated = self.collate(all_subjects_loaded)
        return all_subjects_collated
