import os
import json
from glob import glob
import copy
import torch
from typing import Dict, Sequence
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torchio as tio


class ImageDefinition(dict):
    """ Defines an image or label map that will be loaded as a tio.Image
    Args:
        name: Key to the image in the tio.Subject dictionary
        glob_pattern: glob file pattern to match in the subject folder when loading this image
            see https://en.wikipedia.org/wiki/Glob_(programming)
        label_names: Dictionary that is optionally specified for label maps.
            Used to associate a name to each integer in a label map.
        kwargs: Optional fields to be added to the tio.Image
    """
    def __init__(self, name: str, glob_pattern: str, label_names: Dict[str, int] = None, **kwargs):
        super().__init__()
        self["name"] = name
        self["glob_pattern"] = glob_pattern
        if label_names:
            self["label_names"] = label_names
        self.update(kwargs)


class SubjectFolder(Dataset):
    """ A PyTorch Dataset for 3D medical data.

    The data must be organized so that each subject has their own folder within a root directory.
    Then each subjects folder contains all of the images for the subject.
    An attributes.json file can be optionally included in the subject folder to specify per-subject data
    (i.e. subject age, gender, scanner differences, etc.)

    Args:
        path: Path to the root of the subject folder dataset.
        image_defintions: Defines the images in each subject folder to be loaded as a tio.ScalarImage
        label_definitions: Defines the label maps in each subject folder to be loaded as a tio.LabelMap
        collate_images: Images that are grouped on the channel dimension and stored as a tensor when this subject is
            loaded. These tensors are stacked in collate() into (N, C, H, W, D) tensors.
        collate_labels: Labels to be
        transforms: A tio.Transform that is applied to all loaded images and label maps
        require_images: If true, a subject wont be added to the dataset if they are missing any images defined in
            image_definitions or label_definitions.
        include_subjects: If specified, only subjects whose names are in include_subjects are added to the dataset.
        exclude_subjects: If specified, only subjects whose names are not in exclude_subjects are added to the dataset.
        include_attributes: fields in attributes.json that must be present for a subject to be included
            if the field is a list then it must be non-disjoint
        exclude_attributes: fields in attributes.json that must not be found for a subject to be included
            if the field is a list then it must be disjoint
    """
    def __init__(
            self,
            path: str,
            image_definitions: Sequence[ImageDefinition],
            label_definitions: Sequence[ImageDefinition],
            collate_images: Sequence[str] = None,
            collate_labels: Sequence[str] = None,
            transforms: tio.Transform = None,
            require_images: bool = False,
            include_subjects: Sequence[str] = None,
            exclude_subjects: Sequence[str] = None,
            include_attributes: Dict = None,
            exclude_attributes: Dict = None
    ):
        self.path = path
        self.image_definitions = image_definitions
        self.label_definitions = label_definitions
        self.collate_images = collate_images
        self.collate_labels = collate_labels
        self.require_images = require_images
        self.transforms = transforms
        self.include_subjects = include_subjects
        self.exclude_subjects = exclude_subjects
        self.include_attributes = include_attributes
        self.exclude_attributes = exclude_attributes

        # Loops through all subjects in the directory
        self.all_subjects = []
        self.subject_names = os.listdir(path)
        for subject_name in self.subject_names:

            # The subject_data dictionary will be used to initialize the tio.Subject
            subject_folder = f"{path}/{subject_name}/"
            subject_data = dict(name=subject_name, folder=subject_folder)

            # Load an optional attributes file in the subject folder
            attributes_file = "attributes.json"
            if attributes_file in os.listdir(subject_folder):
                with open(f"{subject_folder}/{attributes_file}") as f:
                    subject_data.update(json.load(f))

            # Load each image_def as a tio.ScalarImage
            if self.image_definitions:
                for image_def in self.image_definitions:
                    matching_files = glob(subject_folder + image_def["glob_pattern"])
                    if len(matching_files) == 0:
                        continue
                    subject_data[image_def["name"]] = tio.ScalarImage(matching_files[0], **image_def)

            # Load each label_def as a tio.LabelMap
            if self.label_definitions:
                for label_def in self.label_definitions:
                    matching_files = glob(subject_folder + label_def["glob_pattern"])
                    if len(matching_files) == 0:
                        continue
                    subject_data[label_def["name"]] = tio.LabelMap(*matching_files, **label_def)

            self.all_subjects.append(tio.Subject(**subject_data))

        # Helper function, converts things to python sets
        def as_set(x):
            if x is None or type(x) in (str, int, bool, float):
                return {x}
            return set(x)

        # Divide 'all_subjects' into 'subjects' and 'excluded_subjects' based on the inclusion/exclusion criteria
        self.subjects = []
        self.excluded_subjects = []
        for subject in self.all_subjects:

            # Handle the include_subjects and exclude_subjects params
            if self.include_subjects is not None and subject.name not in self.include_subjects:
                self.excluded_subjects.append(subject)
                continue
            if self.exclude_subjects is not None and subject.name in self.exclude_subjects:
                self.excluded_subjects.append(subject)
                continue

            # If require_images is true, then exclude subjects that are missing any image_definition or label_definition
            missing_image = False
            if require_images:
                for image_def in list(self.image_definitions) + list(self.label_definitions):
                    if image_def["name"] not in subject:
                        missing_image = True
                        break
            if missing_image:
                self.excluded_subjects.append(subject)
                continue

            # Exclude any subjects that do not have a required attribute
            if include_attributes is not None:
                missing_attribute = False
                for attrib_name, attrib_value in include_attributes.items():
                    if as_set(attrib_value).isdisjoint(as_set(subject.get(attrib_name))):
                        missing_attribute = True
                        break
                if missing_attribute:
                    self.excluded_subjects.append(subject)
                    continue

            # Exclude any subjects that have an excluded attribute
            if exclude_attributes is not None:
                invalid_attribute = False
                for attrib_name, attrib_value in exclude_attributes.items():
                    if not as_set(attrib_value).isdisjoint(as_set(subject.get(attrib_name))):
                        invalid_attribute = True
                        break
                if invalid_attribute:
                    self.excluded_subjects.append(subject)
                    continue

            self.subjects.append(subject)

        # name -> subject dictionaries, for looking up subjects by their name
        self.all_subjects_map = {subject.name: subject for subject in self.all_subjects}
        self.subjects_map = {subject.name: subject for subject in self.subjects}

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
        out_dict = {}

        # Stack the (C, W, H, D) images into a (N, C, W, H, D) image i.e. this adds the batch dimension
        for collate_image_name in self.collate_images:
            out_dict[collate_image_name] = torch.stack([subject[collate_image_name].data for subject in batch])

        # Stack the (1, W, H, D) labels into a (N, W, H, D) label
        # TODO: Switch or add support for multi-channel (N, C, W, H, D) labels. This could be useful for nested labels.
        for collate_image_name in self.collate_labels:
            out_dict[collate_image_name] = torch.cat([subject[collate_image_name].data for subject in batch])

        return out_dict

    def load_and_collate_all_subjects(self):
        all_subjects_loaded = [self[i] for i in range(len(self))]
        all_subjects_collated = self.collate(all_subjects_loaded)
        return all_subjects_collated
