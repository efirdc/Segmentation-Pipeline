import os

import copy
from typing import Dict, Union

from torch.utils.data import Dataset
import torchio as tio

from .subject_loaders import SubjectLoader
from .subject_filters import SubjectFilter, ComposeFilters


class SubjectFolder(Dataset):
    """ A PyTorch Dataset for 3D medical data.

    Args:
        root: Path to the root of the subject folder dataset.
        subject_path: Path to folder containing subjects, relative to the root.
            Each subject must have their own folder within the subject_path.
        subject_loader: A SubjectLoader pipeline that loads subject data from the subject folders.
        cohorts: An optional dictionary that defines different subject cohorts in this dataset.
            The dictionary keys are cohort names, and the values are ``SubjectFilter``s.
            The active cohort can be set with the ``set_cohort(cohort_name)`` method.
            A special cohort name ``'all'`` may be provided to define a filter that is applied
            to all subjects.
        transforms: Optional ``tio.Transform``s that are applied to each subject.
            This can be a single transformation pipeline, or a dictionary that defines
            a number of pipelines.
            The key ``"default"`` can be used to set a default transformation pipeline
            when no cohort is active.
            If a matching key is in `cohorts`, then that transformation will become
            active when ``set_cohort(cohort_name)`` is called.
            A transformation can also be explicitly set with ``set_transform(transform_name)``.
    """
    def __init__(
            self,
            root: str,
            subject_path: str,
            subject_loader: SubjectLoader,
            cohorts: Dict[str, SubjectFilter] = None,
            transforms: Union[tio.Transform, Dict[str, tio.Transform]] = None,
    ):
        self.root = root
        self.subject_path = os.path.join(self.root, subject_path)
        self.subject_loader = subject_loader
        self.cohorts = {} if cohorts is None else cohorts
        self.transforms = transforms

        self._preloaded = False
        self._pretransformed = False

        # Loops through all subjects in the directory
        subjects = []
        subject_names = os.listdir(self.subject_path)
        for subject_name in subject_names:

            # The subject_data dictionary will be used to initialize the tio.Subject
            subject_folder = os.path.join(self.subject_path, subject_name)
            subject_data = dict(name=subject_name, folder=subject_folder)

            # Apply subject loaders
            self.subject_loader(subject_data)

            # torchio doesn't like to load a subject with no images
            if not any(isinstance(v, tio.Image) for v in subject_data.values()):
                continue

            subjects.append(tio.Subject(**subject_data))

        if "all" in self.cohorts:
            all_filter = self.cohorts['all']
            subjects = all_filter(subjects)

        self.active_cohort = 'all'

        self.all_subjects = None
        self.all_subjects_map = None
        self.subjects = None
        self.subjects_map = None
        self.excluded_subjects = None
        self.transform = None

        self.set_all_subjects(subjects)

    def set_all_subjects(self, subjects):
        subjects.sort(key=lambda subject: subject['name'])
        self.all_subjects = subjects
        self.all_subjects_map = {subject['name']: subject for subject in subjects}
        self.set_cohort(self.active_cohort)

    def set_subjects(self, subjects):
        self.subjects = subjects
        self.subjects_map = {subject['name']: subject for subject in subjects}
        self.excluded_subjects = [subject for subject in self.all_subjects
                                  if subject not in self.subjects]

    def set_cohort(self, cohort: Union[str, SubjectFilter]):
        self.active_cohort = cohort

        if isinstance(cohort, str):
            self.set_transform(cohort)
            if cohort == "all" or cohort is None:
                self.set_subjects(self.all_subjects)
            elif cohort in self.cohorts:
                subject_filter = self.cohorts[cohort]
                self.set_subjects(subject_filter(self.all_subjects))
            else:
                raise ValueError(f"Cohort name {cohort} is not defined in dataset cohorts: {self.cohorts}.")
        if isinstance(cohort, SubjectFilter):
            self.set_transform('default')
            subject_filter = cohort
            self.set_subjects(subject_filter(self.all_subjects))

    def set_transform(self, transform_name):
        if self.transforms is None:
            self.transform = None
        if isinstance(self.transforms, tio.Transform):
            self.transform = self.transforms
        if isinstance(self.transforms, dict):
            if transform_name in self.transforms:
                self.transform = self.transforms[transform_name]
            elif 'default' in self.transforms:
                self.transform = self.transforms['default']
            else:
                self.transform = None

    def get_cohort_dataset(self, cohort: Union[str, SubjectFilter]):
        transforms = self.transforms
        if isinstance(cohort, str):
            subject_filter = self.cohorts[cohort]
            if isinstance(transforms, dict):
                transforms = transforms.copy()
                if cohort in transforms:
                    transforms['default'] = transforms[cohort]
                    del transforms[cohort]
        elif isinstance(cohort, SubjectFilter):
            subject_filter = cohort
        else:
            raise ValueError()

        cohorts = self.cohorts.copy()
        if 'all' in cohorts:
            cohorts['all'] = ComposeFilters(cohorts['all'], subject_filter)
        else:
            cohorts['all'] = subject_filter

        return SubjectFolder(self.root, self.subject_path, self.subject_loader, cohorts, transforms)

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
        if not self._preloaded:
            subject.load()
        if not self._pretransformed and self.transform is not None:
            subject = self.transform(subject)

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
        if self._preloaded:
            return
        self._preloaded = True

        loaded_subjects = []
        for subject in self.all_subjects:
            subject = copy.deepcopy(subject)
            subject.load()
            loaded_subjects.append(subject)
        self.set_all_subjects(loaded_subjects)
        self.set_cohort(self.active_cohort)

    def preload_and_transform_subjects(self):
        if self._pretransformed:
            return

        self.preload_subjects()
        if self.transform is not None:
            self._pretransformed = True
            self.set_all_subjects([self.transform(subject) for subject in self.subjects])

    # TODO: Do this better.
    def load_additional_data(self, path: str, subject_loader: SubjectLoader):

        subject_names = os.listdir(path)
        for subject_name in subject_names:

            subject_folder = os.path.join(path, subject_name)
            subject_data = dict(name=subject_name, folder=subject_folder)

            subject_loader(subject_data)
            del subject_data['name']
            del subject_data['folder']

            self.all_subjects_map[subject_name].update(subject_data)
