from random import Random
from typing import Sequence, Dict, Union, Any

import torchio as tio

from ..utils import as_set, is_sequence, vargs_or_sequence, as_list, auto_str


class SubjectFilter:
    """ Base class for filtering subjects

    Implementations which depend only the attributes of a single subject
    can overwrite the `subject_filter` method which takes a `tio.Subject`
    and returns a bool that is `True` if the subject will be kept and `False`
    if the subject will be filtered.

    For cases like generating train/test splits, overwrite the `apply_filter` method.

    """
    def __call__(
            self,
            *subjects: Union[tio.Subject, Sequence[tio.Subject]]
    ):
        subjects = vargs_or_sequence(subjects)
        if is_sequence(subjects) and all(isinstance(subject, tio.Subject) for subject in subjects):
            return self.apply_filter(subjects)
        else:
            raise ValueError("A SubjectFilter can only be applied to a sequence of tio.Subject, "
                             f"not {subjects}")

    def apply_filter(self, subjects: Sequence[tio.Subject]):
        return list(filter(self.subject_filter, subjects))

    def subject_filter(self, subject: tio.Subject) -> bool:
        raise NotImplementedError()

    def __sub__(self, other):
        assert other is SubjectFilter
        return ComposeFilters(self, NegateFilter(other))

    def __neg__(self):
        return NegateFilter(self)

    def __invert__(self):
        return NegateFilter(self)

    def __repr__(self):
        return auto_str(self)


class RequireAttributes(SubjectFilter):
    """ Filters subjects which do not have the required attributes

    Args:
        attributes: Defines the required attributes for subjects. It can be one of:

            - A sequence of keys that are required to be in the subject dictionary.

            - A dictionary that defines both required keys and values in the subject dictionary.

    Examples:
        >>> # Requires that subjects have T1w, T2w images and anatomical labels
        >>> RequireAttributes(['t1', 't2', 'anatomical_label']),
        >>>
        >>> # Requires that subjects are between 10 and 34 years old and have no disease
        >>> RequireAttributes({'age': range(10, 35), "disease": False})
    """
    def __init__(
            self,
            attributes: Union[Sequence[str], Dict[str, Any]]
    ):
        self.attributes = attributes

    def subject_filter(self, subject):
        if isinstance(self.attributes, list):
            return all(attribute in subject for attribute in self.attributes)
        if isinstance(self.attributes, dict):
            if any(attribute not in subject for attribute in self.attributes.keys()):
                return False
            return all(
                not as_set(attrib_value).isdisjoint(as_set(subject.get(attrib_name)))
                for attrib_name, attrib_value in self.attributes.items()
            )


class ForbidAttributes(SubjectFilter):
    """ Filters subjects which have forbidden attributes

    Args:
        attributes: Defines the forbidden attributes for subjects. It can be one of:

            - A sequence of keys that are forbidden to be in the subject dictionary.

            - A dictionary that defines forbidden values for keys in the subject dictionary.
              Note that the keys themselves are not forbidden. The constraint is on the values only.

    Examples:
        >>> # Forbid subjects that have an anatomical label
        >>> ForbidAttributes(['anatomical_label']),
        >>>
        >>> # Forbid subjects that are in the validation set
        >>> validation_subjects = ["subject_05", "subject_08"]
        >>> ForbidAttributes({'name': validation_subjects})
    """
    def __init__(
            self,
            attributes: Union[Sequence[str], Dict[str, Any]]
    ):
        self.attributes = attributes

    def subject_filter(self, subject):
        if isinstance(self.attributes, list):
            return not any(attribute in subject for attribute in self.attributes)
        if isinstance(self.attributes, dict):
            attributes_in_subject = {
                attrib_name: attrib_value for attrib_name, attrib_value in self.attributes.items()
                if attrib_name in subject
            }
            return all(
                as_set(attrib_value).isdisjoint(as_set(subject.get(attrib_name)))
                for attrib_name, attrib_value in attributes_in_subject.items()
            )


class ComposeFilters(SubjectFilter):
    """ Combines `SubjectFilter`s using a logical and.

    Args:
        filters: A sequence of filters. If any filter returns False the subject will be filtered.
    """
    def __init__(self, *filters: Union[SubjectFilter, Sequence[SubjectFilter]]):
        self.filters = vargs_or_sequence(filters)

    def apply_filter(self, subjects):
        groups = [
            subject_filter(subjects)
            for subject_filter in self.filters
        ]
        subjects = [
            subject for subject in subjects
            if all(subject in group for group in groups)
        ]
        return subjects


class AnyFilter(SubjectFilter):
    """ Combines `SubjectFilter`s using a logical or.

    Args:
        filters: A sequence of filters. If any filter returns True the subject will not be filtered.
    """
    def __init__(self, *filters: Union[SubjectFilter, Sequence[SubjectFilter]]):
        self.filters = vargs_or_sequence(filters)

    def apply_filter(self, subjects):
        if len(self.filters) == 0:
            return subjects
        groups = [
            subject_filter(subjects)
            for subject_filter in self.filters
        ]
        subjects = [
            subject for subject in subjects
            if any(subject in group for group in groups)
        ]
        return subjects


class NegateFilter(SubjectFilter):
    """ Negates the provided `SubjectFilter`.

    Args:
        filter: A filter to be negated.
    """
    def __init__(self, filter: SubjectFilter):
        self.filter = filter

    def apply_filter(self, subjects):
        remove_subjects = self.filter(subjects)
        subjects = [
            subject for subject in subjects
            if subject not in remove_subjects
        ]
        return subjects


class RandomFoldFilter(SubjectFilter):
    """Splits subjects into folds in a deterministic random way.

    Adds an int property ``"fold"`` to the subjects the first time this filter is applied.
    Subsequent fold filters do not re-assign the fold, they only select subjects based on
    the previously assigned fold.

    Args:
        num_folds: Subjects are split into this many evenly sized folds.
        selection: Subjects not in these folds are filtered (0 indexed)
        seed: Seed for generating random folds.
    """
    def __init__(
            self,
            num_folds: int,
            selection: Union[int, Sequence[int]],
            seed: int = 0,
    ):
        self.num_folds = num_folds
        self.selection = as_list(selection)
        self.seed = seed

        assert all(0 <= sel < self.num_folds for sel in self.selection)

    def apply_filter(self, subjects):
        folds_assigned = any('fold' in subject for subject in subjects)

        if not folds_assigned:
            fold_ids = [i % self.num_folds for i in range(len(subjects))]
            Random(self.seed).shuffle(fold_ids)

            for i in range(len(subjects)):
                subjects[i]['fold'] = fold_ids[i]

        subjects = [
            subject for subject in subjects
            if 'fold' in subject and subject['fold'] in self.selection
        ]

        return subjects




