from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union
from glob import glob
import pathlib
import json
import copy

import pandas as pd
import numpy as np
import torch

from ..utils import auto_str, vargs_or_sequence


class SubjectLoader(ABC):
    """ Abstract class for loading subject data.

    All subclasses must overwrite the `__call__` method,
    which takes a `subject_data` dictionary which contains the
    a subject's `"name"`, `"folder"` and any other properties
    previously loaded by another `SubjectLoader`.

    """
    @abstractmethod
    def __call__(self, subject_data):
        raise NotImplementedError()

    def __repr__(self):
        return auto_str(self)


class AttributeLoader(SubjectLoader):
    """Loads subject attribute data from a csv, xlsx, or json file.

    A file is assumed to be csv if the extension is unknown.

    Args:
        glob_pattern: glob file pattern to match data files relative to a subject's folder.
            see https://en.wikipedia.org/wiki/Glob_(programming)
            If the glob pattern matches multiple files, then they are all loaded.
        multi_subject: Set to true if the attribute file has data for multiple subjects.
            In this case the first column of the csv or first key in the json file must be the subjects name.
        uniform: Set to true if the glob pattern points to the same file for all subjects.
            The file will be cached for efficiency.
        belongs_to: A key to an existing dictionary in the subject.
            The loaded attributes will be added to this dictionary instead of the subject root.
            This can be used to add properties to tio.Image, e.g. label values and colors for a label map
    """
    def __init__(
            self,
            glob_pattern: str,
            multi_subject: bool = False,
            uniform: bool = False,
            belongs_to: str = None
    ):
        self.glob_pattern = glob_pattern
        self.multi_subject = multi_subject
        self.uniform = uniform
        self.belongs_to = belongs_to

        self.uniform_cache = {}

    def __call__(self, subject_data):
        subject_folder = subject_data['folder']
        matching_files = glob(f"{subject_folder}/{self.glob_pattern}")

        for matching_file in matching_files:

            data = self.load_file(matching_file)

            if self.multi_subject:
                if subject_data['name'] not in data:
                    continue
                data = data[subject_data['name']]

            if self.belongs_to is not None:
                subject_data[self.belongs_to].update(data)
            else:
                subject_data.update(data)

    def load_file(self, file_path):

        if self.uniform and file_path in self.uniform_cache:
            return self.uniform_cache[file_path]

        extension = pathlib.Path(file_path).suffix

        if extension == ".json":
            with open(file_path) as f:
                data = json.load(f)
        else:
            if extension == ".xlsx":
                df = pd.read_excel(file_path, index_col=0)
            else:
                df = pd.read_csv(file_path, index_col=0)
            data = df.to_dict()

        if self.uniform:
            self.uniform_cache[file_path] = data

        return data


class ImageLoader(SubjectLoader):
    """ Loads an image as a tio.ScalarImage or tio.LabelMap

    Args:
        glob_pattern: glob file pattern to match in the subject folder when loading this image
            see https://en.wikipedia.org/wiki/Glob_(programming)
            If the glob pattern matches multiple images
            then they are concatenated on the channel axis.
        image_name: Key for the image in the tio.Subject dictionary
        image_constructor: Either a tio.ScalarImage for floating point data,
            or a tio.LabelMap for integer data.
        uniform: Set to true if the glob pattern points to the same image for all subjects.
            The image will be cached for efficiency.
        kwargs: Optional fields to be added to the image
    """
    def __init__(
            self,
            glob_pattern: str,
            image_name: str,
            image_constructor: Callable,
            uniform: bool = False,
            **kwargs
    ):
        self.image_name = image_name
        self.image_constructor = image_constructor
        self.glob_pattern = glob_pattern
        self.uniform = uniform
        self.kwargs = kwargs

        self.cache = None

    def __call__(self, subject_data):
        if self.uniform:
            subject_data[self.image_name] = copy.deepcopy(self.cached_image)
            return

        subject_folder = subject_data['folder']
        matching_files = glob(f"{subject_folder}/{self.glob_pattern}")
        if len(matching_files) == 0:
            return

        new_image = self.image_constructor(*matching_files, **self.kwargs)
        if self.uniform:
            self.cached_image = new_image
            new_image = copy.deepcopy(new_image)
        subject_data[self.image_name] = new_image


class ComposeLoaders(SubjectLoader):
    """ Composes a sequence of `SubjectLoader`s together.

    The loaders are applied in iteration order, making each

    Args:
        loaders: A sequence of `SubjectLoader`s
    """
    def __init__(self, *loaders: Union[SubjectLoader, Sequence[SubjectLoader]]):
        self.loaders = vargs_or_sequence(loaders)

    def __call__(self, subject_data):
        for loader in self.loaders:
            loader(subject_data)


class TensorLoader(SubjectLoader):
    """Loads subject tensor data from a space delimited text file.

    Args:
        tensor_name: Key for the Tensor in the dictionary
        glob_pattern: glob file pattern to match data files relative to a subject's folder.
            see https://en.wikipedia.org/wiki/Glob_(programming)
            If the glob pattern matches multiple files, then they are all loaded.
        uniform: Set to true if the glob pattern points to the same file for all subjects.
            The file will be cached for efficiency.
        belongs_to: A key to an existing dictionary in the subject.
            The loaded attributes will be added to this dictionary instead of the subject root.
            This can be used to add properties to tio.Image, e.g. label values and colors for a label map
    """
    def __init__(
            self,
            glob_pattern: str,
            tensor_name: str,
            uniform: bool = False,
            belongs_to: str = None
    ):
        self.glob_pattern = glob_pattern
        self.tensor_name = tensor_name
        self.uniform = uniform
        self.belongs_to = belongs_to

        self.uniform_cache = {}

    def __call__(self, subject_data):
        subject_folder = subject_data['folder']
        matching_files = glob(f"{subject_folder}/{self.glob_pattern}")

        if len(matching_files) > 1:
            raise RuntimeError(f"More than one {self.tensor_name} file found in {subject_folder}/{self.glob_pattern}")

        for matching_file in matching_files:

            data = self.load_file(matching_file)

            if self.belongs_to is not None:
                subject_data[self.belongs_to].update(data)
            else:
                subject_data.update(data)

    def load_file(self, file_path):

        if self.uniform and file_path in self.uniform_cache:
            return self.uniform_cache[file_path]


        data = dict()
        data[self.tensor_name] = torch.from_numpy(np.loadtxt(file_path, delimiter=' '))

        if self.uniform:
            self.uniform_cache[file_path] = data

        return data